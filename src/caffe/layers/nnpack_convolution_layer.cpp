#include "nnpack.h"

#include "caffe/layers/nnpack_convolution_layer.hpp"

namespace caffe {

template <typename Dtype>
void NNPackConvolutionLayer<Dtype>::compute_output_shape() {
  const int* kernel_shape_data = this->kernel_shape_.cpu_data();
  const int* stride_data = this->stride_.cpu_data();
  const int* pad_data = this->pad_.cpu_data();
  const int* dilation_data = this->dilation_.cpu_data();
  this->output_shape_.clear();
  for (int i = 0; i < this->num_spatial_axes_; ++i) {
    // i + 1 to skip channel axis
    const int input_dim = this->input_shape(i + 1);
    const int kernel_extent = dilation_data[i] * (kernel_shape_data[i] - 1) + 1;
    const int output_dim =
        (input_dim + 2 * pad_data[i] - kernel_extent) / stride_data[i] + 1;
    this->output_shape_.push_back(output_dim);
  }
}

template <typename Dtype>
enum nnp_operation_status caffe_nnp_forward_convolution(
    enum nnp_convolution_algorithm algorithm,
    size_t minibatch_size,
    size_t input_channels,
    size_t output_channels,
    struct nnp_size input_size,
    struct nnp_size kernel_size,
    struct nnp_padding padding,
    const Dtype input[],
    const Dtype kernel[],
    const Dtype bias[],
    Dtype output[]);

template <>
enum nnp_operation_status caffe_nnp_forward_convolution<double>(
    enum nnp_convolution_algorithm algorithm,
    size_t minibatch_size,
    size_t input_channels,
    size_t output_channels,
    struct nnp_size input_size,
    struct nnp_size kernel_size,
    struct nnp_padding padding,
    const double input[],
    const double kernel[],
    const double bias[],
    double output[]) {
  return nnp_operation_status_unsupported_algorithm;
}

template <>
enum nnp_operation_status caffe_nnp_forward_convolution<float>(
    enum nnp_convolution_algorithm algorithm,
    size_t minibatch_size,
    size_t input_channels,
    size_t output_channels,
    struct nnp_size input_size,
    struct nnp_size kernel_size,
    struct nnp_padding padding,
    const float input[],
    const float kernel[],
    const float bias[],
    float output[]) {
  return nnp_forward_convolution(algorithm,
                                 minibatch_size,
                                 input_channels,
                                 output_channels,
                                 input_size,
                                 kernel_size,
                                 padding,
                                 input,
                                 kernel,
                                 bias,
                                 output,
                                 nullptr,
                                 nullptr);
}

template <>
bool NNPackConvolutionLayer<float>::is_supported() {
  static enum nnp_operation_status nnpack_status = nnp_initialize();
  return nnpack_status == nnp_operation_status_success;
}

template <>
bool NNPackConvolutionLayer<double>::is_supported() {
  return false;
}

template <typename Dtype>
void NNPackConvolutionLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  CHECK(this->bias_term_);
  const Dtype* bias = this->blobs_[1]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    const size_t batch_size = bottom[i]->num();
    const size_t input_channels = bottom[i]->channels();
    const size_t output_channels = top[i]->channels();
    const nnp_size input_size = {static_cast<size_t>(bottom[i]->width()),
                                 static_cast<size_t>(bottom[i]->height())};
    const nnp_size kernel_size = {
        .width = static_cast<size_t>(this->blobs_[0]->width()),
        .height = static_cast<size_t>(this->blobs_[0]->height())};
    const nnp_padding padding = {
        .top = static_cast<size_t>(this->pad_.cpu_data()[0]),
        .right = static_cast<size_t>(this->pad_.cpu_data()[1]),
        .bottom = static_cast<size_t>(this->pad_.cpu_data()[0]),
        .left = static_cast<size_t>(this->pad_.cpu_data()[1])};
    auto algorithm = nnp_convolution_algorithm_wt8x8;
    switch (this->layer_param_.nnpack_convolution_param().algorithm()) {
      case NNPACKConvolutionParameter_Algorithm_WINOGRAD: {
        algorithm = nnp_convolution_algorithm_wt8x8;
        break;
      }
      case NNPACKConvolutionParameter_Algorithm_FFT_16x16: {
        algorithm = nnp_convolution_algorithm_ft16x16;
        break;
      }
      case NNPACKConvolutionParameter_Algorithm_FFT_8x8: {
        algorithm = nnp_convolution_algorithm_ft8x8;
        break;
      }
    }

    const auto status =
        caffe_nnp_forward_convolution<Dtype>(algorithm,
                                             batch_size,
                                             input_channels,
                                             output_channels,
                                             input_size,
                                             kernel_size,
                                             padding,
                                             bottom[i]->cpu_data(),
                                             weight,
                                             bias,
                                             top[i]->mutable_cpu_data());
    CHECK_EQ(nnp_operation_status_success, status);
  }
}

template <typename Dtype>
void NNPackConvolutionLayer<Dtype>::Backward_cpu(
    const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  LOG(FATAL) << "Not implemented";
}

INSTANTIATE_CLASS(NNPackConvolutionLayer);

} // namespace caffe
