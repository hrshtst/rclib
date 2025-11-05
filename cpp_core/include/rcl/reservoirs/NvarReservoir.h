#pragma once

#include "rcl/Reservoir.h"

class NvarReservoir : public Reservoir {
public:
    NvarReservoir(); // Add parameters later

    Eigen::MatrixXd& advance(const Eigen::MatrixXd& input) override;
    void resetState() override;
    const Eigen::MatrixXd& getState() const override;

private:
    Eigen::MatrixXd state;
};
