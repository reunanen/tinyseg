#include <dlib/dnn.h>

namespace tinyseg {

constexpr int border_required = 0;

constexpr unsigned long max_class_count = 10;

// Adapted from: http://dlib.net/dnn_introduction2_ex.cpp.html

template <
    int N, 
    template <typename> class BN, 
    int stride, 
    typename SUBNET
    > 
using block  = BN<dlib::con<N,3,3,1,1,dlib::relu<BN<dlib::con<N,3,3,stride,stride,SUBNET>>>>>;

template <
    template <int,template<typename>class,int,typename> class block, 
    int N, 
    template<typename>class BN, 
    typename SUBNET
    >
using residual = dlib::add_prev1<block<N,BN,1,dlib::tag1<SUBNET>>>;

// residual_down creates a network structure like this:
/*
         input from SUBNET
             /     \
            /       \
         block     downsample(using avg_pool)
            \       /
             \     /
           add tensors (using add_prev2 which adds the output of tag2 with avg_pool's output)
                |
              output
*/
template <
    template <int,template<typename>class,int,typename> class block, 
    int N, 
    template<typename>class BN, 
    typename SUBNET
    >
using residual_down = dlib::add_prev2<dlib::avg_pool<2,2,2,2, dlib::skip1<dlib::tag2<block<N,BN,2,dlib::tag1<SUBNET>>>>>>;

template <typename SUBNET> using res       = dlib::relu<residual<block,8,dlib::bn_con,SUBNET>>;
template <typename SUBNET> using ares      = dlib::relu<residual<block,8,dlib::affine,SUBNET>>;
template <typename SUBNET> using res_down  = dlib::relu<residual_down<block,8,dlib::bn_con,SUBNET>>;
template <typename SUBNET> using ares_down = dlib::relu<residual_down<block,8,dlib::affine,SUBNET>>;

using net_type = dlib::loss_multiclass_log_per_pixel<
                            dlib::bn_con<dlib::con<max_class_count, 1, 1, 1, 1,
                            res<res<res<res<
                            dlib::input<dlib::matrix<dlib::rgb_pixel>>
                            >>>>>>>;

// Replace batch normalization layers with affine layers.
using runtime_net_type = dlib::loss_multiclass_log_per_pixel<
                            dlib::bn_con<dlib::con<max_class_count, 1, 1, 1, 1,
                            ares<ares<ares<ares<
                            dlib::input<dlib::matrix<dlib::rgb_pixel>>
                            >>>>>>>;

}