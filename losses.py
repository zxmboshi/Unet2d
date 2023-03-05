import mindspore 
import mindspore.nn as nn 
import mindspore.nn.functional as F 


class Gradient3D(nn.Cell):
    """
    Calculate Gradient Loss 
    Parameters:
        penalty: 'l1' or 'l2' -loss 
    Inputs: 
        flow: displacement field 
    Returns:
        loss: gradient loss 
    Author: Wang Yibo 
    Time: 2021/12/7
    """
    def __init__(self, penalty='l2'):
        super().__init__()
        self.penalty = penalty

    def construct(self, flow):
        x = flow

        dh = mindspore.abs(x[:, :, 1:, :, :] - x[:, :, :-1, :, :])
        dw = mindspore.abs(x[:, :, :, 1:, :] - x[:, :, :, :-1, :])
        dd = mindspore.abs(x[:, :, :, :, 1:] - x[:, :, :, :, :-1])

        if self.penalty == 'l2':
            dh = dh ** 2 
            dw = dw ** 2 
            dd = dd ** 2 
            
        loss = (mindspore.mean(dh) + mindspore.mean(dw) + mindspore.mean(dd)) / 3.0 
        return loss 


class CrossCorrelation3D(nn.Cell):
    """
    Calculate Local Normalized Cross Correlation Loss (LNCC)
    Parameters:
        in_ch: in channels default = 1 (gray volume)
        kernel: conv kernel default = (9, 9, 9)
    Inputs:
        y_true: true volume  
        y_pred: pred volume
    Returns:
        loss: LNCC loss 
    Author: Wang Yibo
    Time: 2021/12/7
    """
    def __init__(self, in_ch=1, kernel=(9, 9, 9)):
        super().__init__()
        self.filt = mindspore.ones((1, in_ch, kernel[0], kernel[1], kernel[2])).cuda()
        self.padding = (int((kernel[0] - 1) / 2), int((kernel[1] - 1) / 2), int((kernel[2] - 1) / 2))
        self.k_sum = kernel[0] * kernel[1] * kernel[2]
 
    def construct(self, y_true, y_pred):
        I = y_true 
        J = y_pred

        II = I * I 
        JJ = J * J 
        IJ = I * J

        I_sum = F.conv3d(I, self.filt, stride=1, padding=self.padding)
        J_sum = F.conv3d(J, self.filt, stride=1, padding=self.padding)
        II_sum = F.conv3d(II, self.filt, stride=1, padding=self.padding)
        JJ_sum = F.conv3d(JJ, self.filt, stride=1, padding=self.padding)
        IJ_sum = F.conv3d(IJ, self.filt, stride=1, padding=self.padding)

        I_u = I / self.k_sum
        J_u = J / self.k_sum

        cross = IJ_sum - I_sum * J_u - J_sum * I_u + I_u * J_u * self.k_sum
        I_var = II_sum - 2 * I_sum * I_u + I_u * I_u * self.k_sum
        J_var = JJ_sum - 2 * J_sum * J_u + J_u * J_u * self.k_sum

        top = cross * cross
        bottom = I_var * J_var + 1e-5 

        loss = top / bottom
        return -mindspore.mean(loss)
