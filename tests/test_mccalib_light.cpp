#include <filesystem>
#include <string>
#include <vector>

#include <boost/test/unit_test.hpp>

#include <McCalib.hpp>

namespace {

std::filesystem::path makeMinimalConfigFile(const int number_of_boards = 1) {
  const std::filesystem::path config_path =
      std::filesystem::temp_directory_path() / "mc_calib_light_test_config.yml";

  cv::FileStorage fs(config_path.string(), cv::FileStorage::WRITE);
  fs << "number_camera" << 1;
  fs << "number_board" << number_of_boards;
  fs << "refine_corner" << true;
  fs << "min_perc_pts" << 0.2;
  fs << "number_x_square" << 5;
  fs << "number_y_square" << 7;
  fs << "resolution_x" << 500;
  fs << "resolution_y" << 500;
  fs << "square_size" << 0.03;
  fs << "root_path" << config_path.parent_path().string();
  fs << "cam_prefix"
     << "cam_";
  fs << "ransac_threshold" << 5.0;
  fs << "number_iterations" << 5;
  fs << "distortion_model" << 0;

  const std::vector<int> empty_int_vec;
  const std::vector<double> empty_double_vec;
  fs << "distortion_per_camera" << empty_int_vec;
  fs << "boards_index" << empty_int_vec;

  fs << "length_square" << 0.03;
  fs << "length_marker" << 0.02;
  fs << "save_path" << config_path.parent_path().string();
  fs << "camera_params_file_name"
     << "camera_params_test.yml";
  fs << "cam_params_path"
     << "None";
  fs << "keypoints_path"
     << "None";
  fs << "save_reprojection" << 0;
  fs << "save_detection" << 0;
  fs << "square_size_per_board" << empty_double_vec;
  fs << "number_x_square_per_board" << empty_int_vec;
  fs << "number_y_square_per_board" << empty_int_vec;
  fs << "resolution_x_per_board" << empty_int_vec;
  fs << "resolution_y_per_board" << empty_int_vec;
  fs << "he_approach" << 0;
  fs << "fix_intrinsic" << 0;
  fs.release();

  return config_path;
}

} // namespace

BOOST_AUTO_TEST_SUITE(CheckMcCalibLight)

BOOST_AUTO_TEST_CASE(CheckCalibrationCtorInvalidPathKeepsStructuresEmpty) {
  const std::filesystem::path invalid_cfg =
      std::filesystem::temp_directory_path() /
      "definitely_missing_mc_calib_cfg.yml";
  std::filesystem::remove(invalid_cfg);

  McCalib::Calibration calib(invalid_cfg);

  BOOST_CHECK_EQUAL(calib.cams_.size(), 0);
  BOOST_CHECK_EQUAL(calib.boards_3d_.size(), 0);
}

BOOST_AUTO_TEST_CASE(CheckCalibrationCtorMinimalConfigBuildsCameraAndBoard) {
  const std::filesystem::path cfg = makeMinimalConfigFile();

  McCalib::Calibration calib(cfg);

  BOOST_CHECK_EQUAL(calib.cams_.size(), 1);
  BOOST_CHECK_EQUAL(calib.boards_3d_.size(), 1);
  BOOST_CHECK_EQUAL(calib.nb_camera_, 1u);
  BOOST_CHECK_EQUAL(calib.nb_board_, 1u);
}

BOOST_AUTO_TEST_CASE(CheckComputeDistanceBetweenPointsNonEmptyAndEmpty) {
  const std::filesystem::path cfg = makeMinimalConfigFile();
  McCalib::Calibration calib(cfg);

  const std::vector<cv::Point2f> obj_pts_2d = {{0.0f, 0.0f}, {3.0f, 4.0f}};
  const std::vector<cv::Point2f> repro_pts = {{0.0f, 4.0f}, {0.0f, 0.0f}};

  const cv::Mat error_list =
      calib.computeDistanceBetweenPoints(obj_pts_2d, repro_pts);
  BOOST_REQUIRE_EQUAL(error_list.rows, 2);
  BOOST_REQUIRE_EQUAL(error_list.cols, 1);
  BOOST_CHECK_SMALL(
      static_cast<double>(std::abs(error_list.at<float>(0) - 4.0f)), 1e-6);
  BOOST_CHECK_SMALL(
      static_cast<double>(std::abs(error_list.at<float>(1) - 5.0f)), 1e-6);

  const std::vector<cv::Point2f> empty_pts;
  const cv::Mat empty_error_list =
      calib.computeDistanceBetweenPoints(empty_pts, empty_pts);
  BOOST_CHECK_EQUAL(empty_error_list.rows, 0);
}

BOOST_AUTO_TEST_CASE(CheckComputeAvgReprojectionErrorEmptyState) {
  const std::filesystem::path cfg = makeMinimalConfigFile();
  McCalib::Calibration calib(cfg);

  BOOST_CHECK_SMALL(std::abs(calib.computeAvgReprojectionError()), 1e-12);
}

BOOST_AUTO_TEST_CASE(CheckMergeAllObjectObsInitializesEachObjectOnce) {
  const std::filesystem::path cfg = makeMinimalConfigFile(2);
  McCalib::Calibration calib(cfg);

  const std::array<int, 3> color = {0, 0, 0};
  const auto camera = calib.cams_.at(0);
  const std::vector<cv::Point2f> points = {{0.0f, 0.0f}};
  const std::vector<int> point_ids = {0};
  const auto board_obs_0 = std::make_shared<McCalib::BoardObs>(
      0, 0, 0, points, point_ids, camera, calib.boards_3d_.at(0));
  const auto board_obs_1 = std::make_shared<McCalib::BoardObs>(
      0, 0, 1, points, point_ids, camera, calib.boards_3d_.at(1));
  const auto camera_obs = std::make_shared<McCalib::CameraObs>(board_obs_0);
  camera_obs->insertNewBoard(board_obs_1);
  const auto frame = std::make_shared<McCalib::Frame>(0, 0, "frame.png");
  frame->insertNewBoard(board_obs_0);
  frame->insertNewBoard(board_obs_1);
  frame->insertNewCamObs(camera_obs);
  calib.cams_obs_[{0, 0}] = camera_obs;
  calib.frames_[0] = frame;

  for (int object_id = 0; object_id < 2; ++object_id) {
    auto object =
        std::make_shared<McCalib::Object3D>(1, object_id, object_id, color);
    object->boards_[object_id] = calib.boards_3d_.at(object_id);
    object->pts_board_2_obj_[{object_id, 0}] = 0;
    calib.object_3d_[object_id] = object;
  }

  calib.mergeAllObjectObs();

  BOOST_CHECK_EQUAL(calib.object_observations_.size(), 2);
  for (const auto &object : calib.object_3d_)
    BOOST_CHECK_EQUAL(object.second->object_observations_.size(), 1);
  BOOST_CHECK_EQUAL(camera_obs->object_observations_.size(), 2);
  BOOST_CHECK_EQUAL(frame->object_observations_.size(), 2);
  BOOST_CHECK_EQUAL(camera->object_observations_.size(), 2);
}

BOOST_AUTO_TEST_CASE(CheckBoardExtractionWithNoImagesKeepsObservationsEmpty) {
  const std::filesystem::path cfg = makeMinimalConfigFile();
  std::filesystem::create_directories(std::filesystem::temp_directory_path() /
                                      "cam_001");
  McCalib::Calibration calib(cfg);

  calib.boardExtraction();

  BOOST_CHECK_EQUAL(calib.board_observations_.size(), 0);
  BOOST_CHECK_EQUAL(calib.frames_.size(), 0);
}

BOOST_AUTO_TEST_CASE(CheckFinalPoseExportPreservesFrameAndGroupIdentity) {
  const auto config = makeMinimalConfigFile();
  McCalib::Calibration calib(config);
  calib.frames_.clear();
  calib.cam_group_.clear();
  std::vector<std::shared_ptr<McCalib::Camera>> cameras;
  std::vector<std::shared_ptr<McCalib::CameraGroupObs>> observations;
  for (int i = 0; i < 2; ++i) {
    auto camera = std::make_shared<McCalib::Camera>(i, 0);
    camera->im_cols_ = 640;
    camera->im_rows_ = 480;
    camera->setIntrinsics(cv::Mat::eye(3, 3, CV_64F),
                          cv::Mat::zeros(1, 5, CV_64F));
    cameras.push_back(camera);
    auto group = std::make_shared<McCalib::CameraGroup>(i, i + 3);
    group->insertCamera(camera);
    group->relative_camera_pose_[i] = {0., 0., 0., 0., 0., 0.};
    calib.cam_group_[i + 3] = group;
    auto frame = std::make_shared<McCalib::Frame>(7 + 20 * i, i, "test.png");
    auto observation = std::make_shared<McCalib::CameraGroupObs>(group, true);
    observation->object_pose_[0] = {0., 0., 0., 1. + i, 2., 3.};
    observations.push_back(observation);
    frame->cam_group_observations_[i + 3] = observation;
    calib.frames_[frame->frame_idx_] = frame;
  }
  calib.saveVisualizationData();
  calib.saveCamerasParams();
  cv::FileStorage fs((calib.save_path_ / "visualization_data.yml").string(),
                     cv::FileStorage::READ);
  BOOST_CHECK_EQUAL(static_cast<int>(fs["schema_version"]), 1);
  BOOST_REQUIRE_EQUAL(fs["frames"].size(), 2);
  for (int i = 0; i < 2; ++i) {
    auto frame = fs["frames"][i];
    BOOST_CHECK_EQUAL(static_cast<int>(frame["frame_id"]), 7 + 20 * i);
    auto group = frame["groups"][0];
    BOOST_CHECK_EQUAL(static_cast<int>(group["camera_group_id"]), i + 3);
    cv::Mat pose;
    group["poses"][0]["object_to_group"] >> pose;
    BOOST_CHECK_CLOSE(pose.at<double>(0, 3), 1. + i, 1e-9);
    BOOST_CHECK_CLOSE(pose.at<double>(2, 3), 3., 1e-9);
  }
  cv::FileStorage camera_fs(
      (calib.save_path_ / "camera_params_test.yml").string(),
      cv::FileStorage::READ);
  BOOST_CHECK(!camera_fs["camera_0"].empty());
  BOOST_CHECK(!camera_fs["camera_1"].empty());
  fs.release();
  camera_fs.release();
  std::filesystem::remove(calib.save_path_ / "visualization_data.yml");
  std::filesystem::remove(calib.save_path_ / "camera_params_test.yml");
  std::filesystem::remove(config);
}

BOOST_AUTO_TEST_SUITE_END()
