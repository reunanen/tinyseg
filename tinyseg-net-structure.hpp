#include <dlib/dnn.h>

namespace tinyseg {

constexpr size_t input_image_size = 28;
constexpr unsigned long max_class_count = 10;

#if 0

// from: http://dlib.net/dnn_introduction_ex.cpp.html

// Now let's define the LeNet.  Broadly speaking, there are 3 parts to a network
// definition.  The loss layer, a bunch of computational layers, and then an input
// layer.  You can see these components in the network definition below.  
// 
// The input layer here says the network expects to be given matrix<unsigned char>
// objects as input.  In general, you can use any dlib image or matrix type here, or
// even define your own types by creating custom input layers.
//
// Then the middle layers define the computation the network will do to transform the
// input into whatever we want.  Here we run the image through multiple convolutions,
// ReLU units, max pooling operations, and then finally a fully connected layer that
// converts the whole thing into just 10 numbers.  
// 
// Finally, the loss layer defines the relationship between the network outputs, our 10
// numbers, and the labels in our dataset.  Since we selected loss_multiclass_log it
// means we want to do multiclass classification with our network.   Moreover, the
// number of network outputs (i.e. 10) is the number of possible labels.  Whichever
// network output is largest is the predicted label.  So for example, if the first
// network output is largest then the predicted digit is 0, if the last network output
// is largest then the predicted digit is 9.  
using net_type = dlib::loss_multiclass_log<
    dlib::fc<max_class_count,
    dlib::relu<dlib::fc<84,
    dlib::relu<dlib::fc<120,
    dlib::max_pool<2, 2, 2, 2, dlib::relu<dlib::con<16, 5, 5, 1, 1,
    dlib::max_pool<2, 2, 2, 2, dlib::relu<dlib::con<6, 5, 5, 1, 1,
    dlib::input<dlib::matrix<unsigned char>>
    >>>>>>>>>>>>;
#endif



#if 1

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
using net_type = dlib::loss_multiclass_log<dlib::fc<max_class_count,
                            dlib::avg_pool_everything<
                            res<res<res<res_down<
                            dlib::repeat<9,res, // repeat this layer 9 times
                            res_down<
                            res<
                            dlib::input<dlib::matrix<unsigned char>>
                            >>>>>>>>>>;

// Replace batch normalization layers with affine layers.
using runtime_net_type = dlib::loss_multiclass_log<dlib::fc<max_class_count,
                            dlib::avg_pool_everything<
                            ares<ares<ares<ares_down<
                            dlib::repeat<9,ares,
                            ares_down<
                            ares<
                            dlib::input<dlib::matrix<unsigned char>>
                            >>>>>>>>>>;

// ----------------------------------------------------------------------------------------
#endif

}