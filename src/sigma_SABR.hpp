//
// Created by anatolii on 10/15/22.
//

#ifndef SABR_TORCH_SIGMA_SABR_H
#define SABR_TORCH_SIGMA_SABR_H

#include <torch/torch.h>

    torch::Tensor sigma_SABR(torch::Tensor K = torch::linspace(0.04, 0.11, 25), torch::Tensor p, torch::Tensor S = 0.06, torch::Tensor beta = 0.5, torch::Tensor maturity = 0.5) {
        alpha, nu, rho = p;

    torch::Tensor K = torch::linspace(0.04, 0.11, 25);

    double T = 0.5;
    double alpha = 0.037561;
    double beta = 0.5;
    double rho = 0.100044;
    double nu = 0.573296;
    double beta = 0.5;
    double maturity = 0.5;

        torch::Tensor zeta = nu / alpha * (S * K) * *((1.0 - beta) / 2) * torch::log(S / K);

        torch::Tensor numen = (1.0 + (((1 - beta) * *2 / 24) * alpha * *2 / ((S * K) * *(1 - beta))
                             + (rho * beta * nu * alpha) / (4.0 * (S * K) * *((1 - beta) / 2))
                             + ((2 - 3 * rho * *2) / 24 * nu * *2)) * maturity);

        torch::Tensor denum = ((S * K) * *((1 - beta) / 2) *
                      (1.0 + (1 - beta) * *2 / 24 * torch::log(S / K) * *2
                       + (1 - beta) * *4 / 1920 * torch::log(S / K) * *4));


        return (alpha * numen / denum * zeta / torch::log((torch::sqrt(1.0 - 2 * rho * zeta + zeta * *2
        ) + zeta - rho) / (1.0 - rho)));
    }

#endif //SABR_TORCH_SIGMA_SABR_H
