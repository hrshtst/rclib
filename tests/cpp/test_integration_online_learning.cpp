#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <iostream>
#include <rclib/Model.h>
#include <rclib/readouts/LmsReadout.h>
#include <rclib/readouts/RlsReadout.h>
#include <rclib/reservoirs/RandomSparseReservoir.h>
#include <vector>

// Helper to generate a signal with a frequency change
std::vector<double> generate_switching_sine(int n_steps, int change_point) {
  std::vector<double> signal(n_steps);
  double pi = std::acos(-1.0);
  for (int i = 0; i < n_steps; ++i) {
    double freq = (i < change_point) ? 0.5 : 1.2;
    double t = static_cast<double>(i) * 30.0 / n_steps;
    signal[i] = std::sin(t * 2.0 * pi * freq);
  }
  return signal;
}

double run_online_cpp_experiment(std::shared_ptr<Readout> readout) {
  int n_steps = 600;
  int change_point = 300;
  auto data = generate_switching_sine(n_steps, change_point);

  auto res = std::make_shared<RandomSparseReservoir>(100, 0.9, 0.1, 1.0, 1.0, true, 42);

  Model model;
  model.addReservoir(res);
  model.setReadout(readout);

  // Initial training on stable part
  int init_train_len = 100;
  Eigen::MatrixXd x_init(init_train_len, 1);
  Eigen::MatrixXd y_init(init_train_len, 1);
  for (int i = 0; i < init_train_len; ++i) {
    x_init(i, 0) = data[i];
    y_init(i, 0) = data[i + 1];
  }
  model.fit(x_init, y_init, 10);

  // Online adaptation
  double total_se = 0;
  int count = 0;
  for (int i = init_train_len; i < n_steps - 1; ++i) {
    Eigen::MatrixXd curr_x(1, 1);
    Eigen::MatrixXd curr_y(1, 1);
    curr_x(0, 0) = data[i];
    curr_y(0, 0) = data[i + 1];

    Eigen::MatrixXd pred = model.predictOnline(curr_x);
    double err = pred(0, 0) - curr_y(0, 0);
    total_se += err * err;

    model.getReadout()->partialFit(res->getState(), curr_y);
    count++;
  }
  return total_se / count;
}

TEST_CASE("Integration: Online Learning Adaptation", "[integration][online]") {
  SECTION("LMS Adaptation") {
    auto lms = std::make_shared<LmsReadout>(0.01, true);
    double mse = run_online_cpp_experiment(lms);
    REQUIRE(mse < 0.1);
  }

  SECTION("RLS Adaptation") {
    auto rls = std::make_shared<RlsReadout>(0.99, 1.0, true);
    double mse = run_online_cpp_experiment(rls);
    REQUIRE(mse < 0.05);
  }
}
