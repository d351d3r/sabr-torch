#ifndef SABR_TORCH_LM_H
#define SABR_TORCH_LM_H


class LevenbergMarquad {
public:
    LevenbergMarquad(x,y,func) {
        this.x = x;
        this.y = y;
        this.func = func;

    }

    int iter_n = 0;
    int p = p.detach().clone();
    p = p.requires_grad_(True);
    {
        torch::NoGradGuard no_grad;
        double J = torch.autograd.functional.jacobian(func,p);
    }
    double W = torch.diag(torch.tensor([1 / kwargs.get('sigma') ** 2] * len(x)));
    float lambda_lm = torch.tensor(kwargs.get('lambda_lm'));

    float eps1 = torch.tensor(kwargs.get('eps1'));
    float eps2 = torch.tensor(kwargs.get('eps2'));
    float eps3 = torch.tensor(kwargs.get('eps3')) ;
    float eps4 = torch.tensor(kwargs.get('eps4'));
    float lm_up = torch.tensor(kwargs.get('lm_up'));
    float lm_down = torch.tensor(kwargs.get('lm_down'));


    void broyden_jacobian_update() {
        torch::NoGradGuard no_grad;

        df = func(p + dp) - func(p);
        J += torch.outer(df - torch.mv(J, dp),
                         dp).div(torch.linalg.norm(dp, ord = 2));
    }
    void torch_jacobian_update(int &p) {
        torch::NoGradGuard no_grad;
        J = torch::autograd::functional::jacobian(func, p);

    }

    void solve_for_dp() {
        torch::NoGradGuard no_grad;
        JTW = torch.matmul(torch.transpose(J, 0, 1), W);
        JTWJ = torch.matmul(JTW, J);

        dy = y_data - func(p);
        dp = torch.linalg.solve(JTWJ
                                     + self.lambda_lm * torch.diag(torch.diagonal(JTWJ)),
                                     torch.mv(JTW, dy));
    }


    void chi_2(p) {
        torch::NoGradGuard no_grad;
//        '''
//
//        chi2 = y^T.W.y + 2 * y^T.W . y_hat +  (y-hat)^T.W.y_hat
//
//        '''
        y_hat = self.func(p)

        return torch.dot(self.y_data, torch.mv(self.W, self.y_data)) - 2 * torch.dot(self.y_data,
                                                                                     torch.mv(self.W,
                                                                                              y_hat)) + torch.dot(y_hat,
                                                                                                                  torch.mv(
                                                                                                                          self.W,
                                                                                                                          y_hat))
    }

    void rho() {
        torch::NoGradGuard no_grad;

//    '''
//    rho =  chi2(p) - chi2(p + dp) / (dp)^T . ( lambda * diag(J^T W J).dp + J^T W . dy )
//
//    '''

        dy = self.y_data - self.func(self.p)
        rho = ((self.chi_2(self.p) - self.chi_2(self.p + self.dp))
                .div(torch.dot(self.dp, torch.mv(self.lambda_lm * torch.diag(torch.diagonal(self.JTWJ)), self.dp)
                                        + torch.mv(self.JTW, dy)
                     )
                )
        )

        if rho > self.eps4:
        return True
        else:
        return False
    }
    void update_p(dp) {
        torch::NoGradGuard no_grad;
        p += dp;
    }


    void step(self, closure=None) {

        self.dp = 0

        self.solve_for_dp()

        if (self.rho())
        {
            update_p(dp);
            lambda_lm = torch.maximum(lambda_lm / lm_down, torch.tensor(1e-7));
        }
        else
        lambda_lm = torch.minimum(lambda_lm * lm_up, torch.tensor(1e7));

        if (iter_n % (2 * len(p)) == 0)
            broyden_jacobian_update();
        else {

            p.requires_grad_(true);

            torch_jacobian_update(p);

            iter_n += 1;

            return p;
        }
    }

private:
    int x;
    int y;
    int p;
    int func;
};


#endif //SABR_TORCH_LM_H
