#pragma once

#include "rcl/Readout.h"

class LmsReadout : public Readout {
public:
    LmsReadout(); // Add parameters later

    void fit(const Eigen::MatrixXd& states, const Eigen::MatrixXd& targets) override;
    void partialFit(const Eigen::MatrixXd& state, const Eigen::MatrixXd& target) override;
    Eigen::MatrixXd predict(const Eigen::MatrixXd& states) override;

private:
    Eigen::MatrixXd W_out;
};
