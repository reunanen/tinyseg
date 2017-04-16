#include <dlib/dnn.h>

namespace tinyseg {

constexpr int border_required = 0;

constexpr unsigned long max_class_count = 10;

// Adapted from: http://dlib.net/dnn_introduction2_ex.cpp.html

// ----------------------------------------------------------------------------------------

// Let's start by showing how you can conveniently define large and complex
// networks.  The most important tool for doing this are C++'s alias templates.
// These let us define new layer types that are combinations of a bunch of other
// layers.  These will form the building blocks for more complex networks.

// So let's begin by defining the building block of a residual network (see
// Figure 2 in Deep Residual Learning for Image Recognition by He, Zhang, Ren,
// and Sun).  We are going to decompose the residual block into a few alias
// statements.  First, we define the core block.

// Here we have parameterized the "block" layer on a BN layer (nominally some
// kind of batch normalization), the number of filter outputs N, and the stride
// the block operates at.
template <
    int N, 
    template <typename> class BN, 
    int stride, 
    typename SUBNET
    > 
using block  = BN<dlib::con<N,3,3,1,1,dlib::relu<BN<dlib::con<N,3,3,stride,stride,SUBNET>>>>>;

// Next, we need to define the skip layer mechanism used in the residual network
// paper.  They create their blocks by adding the input tensor to the output of
// each block.  So we define an alias statement that takes a block and wraps it
// with this skip/add structure.

// Note the tag layer.  This layer doesn't do any computation.  It exists solely
// so other layers can refer to it.  In this case, the add_prev1 layer looks for
// the tag1 layer and will take the tag1 output and add it to the input of the
// add_prev1 layer.  This combination allows us to implement skip and residual
// style networks.  We have also set the block stride to 1 in this statement.
// The significance of that is explained next.
template <
    template <int,template<typename>class,int,typename> class block, 
    int N, 
    template<typename>class BN, 
    typename SUBNET
    >
using residual = dlib::add_prev1<block<N,BN,1,dlib::tag1<SUBNET>>>;

// Some residual blocks do downsampling.  They do this by using a stride of 2
// instead of 1.  However, when downsampling we need to also take care to
// downsample the part of the network that adds the original input to the output
// or the sizes won't make sense (the network will still run, but the results
// aren't as good).  So here we define a downsampling version of residual.  In
// it, we make use of the skip1 layer.  This layer simply outputs whatever is
// output by the tag1 layer.  Therefore, the skip1 layer (there are also skip2,
// skip3, etc. in dlib) allows you to create branching network structures.

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



// Now we can define 4 different residual blocks we will use in this example.
// The first two are non-downsampling residual blocks while the last two
// downsample.  Also, res and res_down use batch normalization while ares and
// ares_down have had the batch normalization replaced with simple affine
// layers.  We will use the affine version of the layers when testing our
// networks.
template <typename SUBNET> using res       = dlib::relu<residual<block,8,dlib::bn_con,SUBNET>>;
template <typename SUBNET> using ares      = dlib::relu<residual<block,8,dlib::affine,SUBNET>>;
template <typename SUBNET> using res_down  = dlib::relu<residual_down<block,8,dlib::bn_con,SUBNET>>;
template <typename SUBNET> using ares_down = dlib::relu<residual_down<block,8,dlib::affine,SUBNET>>;



// Now that we have these convenient aliases, we can define a residual network
// without a lot of typing.  Note the use of a repeat layer.  This special layer
// type allows us to type repeat<9,res,SUBNET> instead of
// res<res<res<res<res<res<res<res<res<SUBNET>>>>>>>>>.  It will also prevent
// the compiler from complaining about super deep template nesting when creating
// large networks.
using net_type = dlib::loss_multiclass_log_matrixoutput<
                            dlib::bn_con<dlib::con<max_class_count, 1, 1, 1, 1,
                            res<res<res<res</*
                            dlib::repeat<9,res, // repeat this layer 9 times
                            res<
                            res<*/
                            dlib::input<dlib::matrix<unsigned char>>
                            /*>>>*/>>>>>>>;

// Replace batch normalization layers with affine layers.
using runtime_net_type = dlib::loss_multiclass_log_matrixoutput<
                            dlib::bn_con<dlib::con<max_class_count, 1, 1, 1, 1,
                            ares<ares<ares<ares</*
                            dlib::repeat<9,ares,
                            ares<
                            ares<*/
                            dlib::input<dlib::matrix<unsigned char>>
                            /*>>>*/>>>>>>>;

}