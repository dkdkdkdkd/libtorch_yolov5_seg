#include <iostream>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <chrono>

int main() {
    // Load the YOLO model
    torch::jit::script::Module model = torch::jit::load("../yolov5s-seg.torchscript");
    model.to(at::kCPU);
    // Open the webcam
    cv::VideoCapture cap(1); // 0 is the ID of the default camera
    if (!cap.isOpened()) {
        std::cerr << "Could not open the webcam!" << std::endl;
        return -1;
    }

    cv::Mat frame;
    while (true) {
        auto start_time = std::chrono::high_resolution_clock::now();
        cap >> frame; // Get a new frame from the webcam
        cv::resize(frame, frame, cv::Size(640, 640));
        // Convert the frame to a tensor
        torch::Tensor tensor_image = torch::from_blob(frame.data, {1, frame.rows, frame.cols, 3}, torch::kByte);
        tensor_image = tensor_image.permute({0, 3, 1, 2}); // Change shape to {1, 3, height, width}
        tensor_image = tensor_image.to(at::kFloat).div(255);

        // Perform segmentation
        std::vector<torch::jit::IValue> inputs = {tensor_image};
        auto outputs = model.forward(inputs).toTuple();
        
        at::Tensor main_output = outputs->elements()[0].toTensor(); // Modify this if needed
        
        // Check the shape of the output

        // For the sake of example, let's assume the mask output is at index 1
        at::Tensor mask_output = outputs->elements()[1].toTensor();

        // Assuming you want the mask corresponding to a specific class, for example, class index 0:
        // Convert mask tensor to byte tensor
        at::Tensor person_mask = mask_output[0][3].gt(0.5).to(torch::kU8);  

        // Convert tensor to OpenCV mat without memory sharing
        cv::Mat mask(160, 160, CV_8UC1);
        std::memcpy(mask.data, person_mask.data_ptr(), person_mask.numel() * sizeof(uint8_t));

        // Resize the mask to match frame dimensions
        cv::resize(mask, mask, cv::Size(frame.cols, frame.rows));

        // Create a colored version of the mask
        cv::Mat coloredMask = cv::Mat::zeros(frame.size(), frame.type());

        // Here, we only color the region where mask is ON (i.e., mask value > 0)
        coloredMask.setTo(cv::Scalar(0, 0, 255), mask);  // Setting the masked region to red

        // Overlay mask onto original frame
        double alpha = 0.4;
        cv::addWeighted(coloredMask, alpha, frame, 1 - alpha, 0, frame);

        // Display the results
        cv::imshow("Segmentation", frame);
        if (cv::waitKey(30) >= 0) break;

        auto end_time = std::chrono::high_resolution_clock::now();
        auto elapsed_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        double fps = 1000.0 / elapsed_time;

        // Print FPS
        std::cout << "FPS: " << fps << std::endl;
    }

    return 0;
}

