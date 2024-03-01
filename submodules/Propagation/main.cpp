#include "main.h"
#include "Propagation.h"
#include <iostream>
#include <sstream>

void GenerateSampleList(const std::string &dense_folder, std::vector<Problem> &problems)
{
    std::string cluster_list_path = dense_folder + std::string("/pair.txt");

    problems.clear();

    std::ifstream file(cluster_list_path);

    int num_images;
    file >> num_images;

    for (int i = 0; i < num_images; ++i) {
        Problem problem;
        problem.src_image_ids.clear();
        file >> problem.ref_image_id;

        int num_src_images;
        file >> num_src_images;
        for (int j = 0; j < num_src_images; ++j) {
            int id;
            float score;
            file >> id >> score;
            if (score <= 0.0f) {
                continue;
            }
            problem.src_image_ids.push_back(id);
        }
        problems.push_back(problem);
    }
}

void ProcessProblem(const std::string &dense_folder, const Problem &problem, bool geom_consistency, int patch_size)
{
    // std::cout << "Processing image " << std::setw(8) << std::setfill('0') << problem.ref_image_id << "..." << std::endl;
    cudaSetDevice(1);
    std::stringstream result_path;
    result_path << dense_folder << "/propagated_depth";
    std::string result_folder = result_path.str();
    mkdir(result_folder.c_str(), 0777);
    // std::cout << result_folder << std::endl;

    Propagation pro;
    int temp = pro.GetPatchSize();
    pro.SetPatchSize(patch_size);
    temp = pro.GetPatchSize();
    if (geom_consistency) {
        pro.SetGeomConsistencyParams();
    }
    pro.InuputInitialization(dense_folder, problem);

    pro.CudaSpaceInitialization(dense_folder, problem);
    pro.RunPatchMatch();

    const int width = pro.GetReferenceImageWidth();
    const int height = pro.GetReferenceImageHeight();

    cv::Mat_<float> depths = cv::Mat::zeros(height, width, CV_32FC1);
    cv::Mat_<cv::Vec3f> normals = cv::Mat::zeros(height, width, CV_32FC3);
    cv::Mat_<float> costs = cv::Mat::zeros(height, width, CV_32FC1);

    for (int col = 0; col < width; ++col) {
        for (int row = 0; row < height; ++row) {
            int center = row * width + col;
            float4 plane_hypothesis = pro.GetPlaneHypothesis(center);
            depths(row, col) = plane_hypothesis.w;
            normals(row, col) = cv::Vec3f(plane_hypothesis.x, plane_hypothesis.y, plane_hypothesis.z);
            costs(row, col) = pro.GetCost(center);
        }
    }

    std::string suffix = "/depths.dmb";
    if (geom_consistency) {
        suffix = "/depths_geom.dmb";
    }
    std::string depth_path = result_folder + suffix;
    std::string normal_path = result_folder + "/normals.dmb";
    std::string cost_path = result_folder + "/costs.dmb";
    writeDepthDmb(depth_path, depths);
    writeNormalDmb(normal_path, normals);
    writeDepthDmb(cost_path, costs);
    // std::cout << "Processing image " << std::setw(8) << std::setfill('0') << problem.ref_image_id << " done!" << std::endl;
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cout << "USAGE: Propagation filespath ref_id src_ids patchsize" << std::endl;
        return -1;
    }

    std::string dense_folder = argv[1];
    int ref_id = std::stoi(argv[2]);
    std::string src_ids_str = argv[3];
    int patch_size = std::stoi(argv[4]);

    Problem problem;
    ref_id >> problem.ref_image_id;

    std::stringstream ss(src_ids_str);
    std::string token;
    
    while (getline(ss, token, ' ')) {
        int src_id = std::stoi(token);
        problem.src_image_ids.push_back(src_id);
    }

    bool geom_consistency = false;
    ProcessProblem(dense_folder, problem, geom_consistency, patch_size);

    return 0;
}
