#pragma once

#include <vector>

#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template <typename Dtype>
class NNPackConvolutionLayer : public BaseConvolutionLayer<Dtype> {
 public:
  explicit NNPackConvolutionLayer(const LayerParameter& param);

  virtual inline const char* type() const { return "NNPackConvolution"; }

  static bool is_supported();

  ~NNPackConvolutionLayer();

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);
  virtual void Backward_cpu(const vector<Blob<Dtype>*>& bottom,
                            const vector<bool>& propagate_down,
                            const vector<Blob<Dtype>*>& top);
  virtual inline bool reverse_dimensions() { return false; }
  virtual void compute_output_shape();

 private:
  void* threadpool;
};

}