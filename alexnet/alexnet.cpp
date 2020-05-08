#include <dlib/dnn.h>
#include <iostream>
#include <dlib/data_io.h>

using namespace std;
using namespace dlib;
 
int main(int argc, char** argv)
{
    // Load the MNIST dataset
    std::vector<matrix<unsigned char>> training_images;
    std::vector<unsigned long> training_labels;
    std::vector<matrix<unsigned char>> testing_images;
    std::vector<unsigned long> testing_labels;
    load_mnist_dataset(argv[1], training_images, training_labels, testing_images, testing_labels);


    //  Define the AlexNet network
    using net_type = loss_multiclass_log<
                                fc<10,        
                                dropout<relu<fc<4096,   
                                dropout<relu<fc<4096, 
                                max_pool<3, 3, 2, 2, relu<con<256, 3, 3, 1, 1,  
                                relu<con<384, 3, 3, 1, 1, 
                                relu<con<384, 3, 3, 1, 1,
                                max_pool<3, 3, 2, 2, l2normalize<relu<con<256, 5, 5, 1, 1, 
                                max_pool<3, 3, 2, 2, l2normalize<relu<con<96, 11, 11, 1, 1, 
                                input<matrix<unsigned char>>>>>>>>>>>>>>>>>>>>>>>>>;
    
    // Train the network
    net_type net;
    dnn_trainer<net_type> trainer(net);
    trainer.set_learning_rate(0.01);
    trainer.set_min_learning_rate(0.00001);
    trainer.set_mini_batch_size(128);
    trainer.be_verbose();
    trainer.set_synchronization_file("mnist_sync", std::chrono::seconds(20));
    trainer.train(training_images, training_labels);
    net.clean();
    serialize("mnist_network.dat") << net;

    // Test and visualize
    std::vector<unsigned long> predicted_labels = net(testing_images);
    int num_right = 0;
    int num_wrong = 0;
    for (size_t i = 0; i < testing_images.size(); ++i)
    {
        if (predicted_labels[i] == testing_labels[i])
            ++num_right;
        else
            ++num_wrong;
        
    }
    cout << "Correct: " << num_right << endl;
    cout << "Incorrect: " << num_wrong << endl;
    cout << "Accuracy:  " << num_right / static_cast<double>(num_right + num_wrong) << endl;

}
