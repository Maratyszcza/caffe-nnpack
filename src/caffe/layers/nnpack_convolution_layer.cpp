#include <cstdlib>
#include <cstdint>
#include <cerrno>

#include "nnpack.h"

#include "caffe/layers/nnpack_convolution_layer.hpp"

namespace caffe {

template <typename Dtype>
NNPackConvolutionLayer<Dtype>::NNPackConvolutionLayer(const LayerParameter& param) :
  BaseConvolutionLayer<Dtype>(param)
{
  uint32_t threads = 0;
  char* omp_num_threads = getenv("OMP_NUM_THREADS");
  if (omp_num_threads != nullptr) {
    errno = 0;
    char* omp_num_threads_parsed = omp_num_threads;
    const unsigned long long threads_parsed = strtoul(omp_num_threads, &omp_num_threads_parsed, 10);
    if (*omp_num_threads_parsed != '\0') {
      LOG(FATAL) << "OMP_NUM_THREADS is not a number";
    } else if ((errno != 0) || (threads_parsed > UINT32_MAX)) {
      LOG(FATAL) << "Invalid number of threads";
    }
    threads = uint32_t(threads_parsed);
  }
  threadpool = pthreadpool_create(threads);
}

template <typename Dtype>
NNPackConvolutionLayer<Dtype>::~NNPackConvolutionLayer() {
  pthreadpool_destroy(static_cast<pthreadpool_t>(threadpool));
  threadpool = nullptr;
}

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
nnp_status caffe_nnp_convolution_output(
    nnp_convolution_algorithm algorithm,
    size_t batch_size,
    size_t input_channels,
    size_t output_channels,
    nnp_size input_size,
    nnp_padding input_padding,
    nnp_size kernel_size,
    const Dtype input[],
    const Dtype kernel[],
    const Dtype bias[],
    Dtype output[],
    pthreadpool_t threadpool);

template <>
nnp_status caffe_nnp_convolution_output<double>(
    nnp_convolution_algorithm algorithm,
    size_t batch_size,
    size_t input_channels,
    size_t output_channels,
    nnp_size input_size,
    nnp_padding input_padding,
    nnp_size kernel_size,
    const double input[],
    const double kernel[],
    const double bias[],
    double output[],
    pthreadpool_t threadpool) {
  return nnp_status_unsupported_algorithm;
}

template <>
nnp_status caffe_nnp_convolution_output<float>(
    nnp_convolution_algorithm algorithm,
    size_t batch_size,
    size_t input_channels,
    size_t output_channels,
    nnp_size input_size,
    nnp_padding input_padding,
    nnp_size kernel_size,
    const float input[],
    const float kernel[],
    const float bias[],
    float output[],
    pthreadpool_t threadpool) {
  return nnp_convolution_output(algorithm,
    batch_size, input_channels, output_channels,
    input_size, input_padding,
    kernel_size,
    input, kernel, bias, output,
    threadpool,
    nullptr);
}

template <>
bool NNPackConvolutionLayer<float>::is_supported() {
  static enum nnp_status nnpack_status = nnp_initialize();
  return nnpack_status == nnp_status_success;
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

    nnp_size input_size;
    input_size.width = static_cast<size_t>(bottom[i]->width());
    input_size.height = static_cast<size_t>(bottom[i]->height());

    nnp_padding input_padding;
    input_padding.top = input_padding.bottom =
      static_cast<size_t>(this->pad_.cpu_data()[0]);
    input_padding.left = input_padding.right =
      static_cast<size_t>(this->pad_.cpu_data()[1]);

    nnp_size kernel_size;
    kernel_size.width = static_cast<size_t>(this->blobs_[0]->width());
    kernel_size.height = static_cast<size_t>(this->blobs_[0]->height());

    auto algorithm = nnp_convolution_algorithm_auto;
    switch (this->layer_param_.nnpack_convolution_param().algorithm()) {
      case NNPACKConvolutionParameter_Algorithm_AUTO:
        algorithm = nnp_convolution_algorithm_auto;
        break;
      case NNPACKConvolutionParameter_Algorithm_WINOGRAD:
        algorithm = nnp_convolution_algorithm_wt8x8;
        break;
      case NNPACKConvolutionParameter_Algorithm_FFT_16x16:
        algorithm = nnp_convolution_algorithm_ft16x16;
        break;
      case NNPACKConvolutionParameter_Algorithm_FFT_8x8:
        algorithm = nnp_convolution_algorithm_ft8x8;
        break;
    }

    const nnp_status status = caffe_nnp_convolution_output<Dtype>(algorithm,
      batch_size, input_channels, output_channels,
      input_size, input_padding,
      kernel_size,
      bottom_data, weight, bias, top_data,
      static_cast<pthreadpool_t>(threadpool));
    CHECK_EQ(nnp_status_success, status);
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
