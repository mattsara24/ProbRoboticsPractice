import math

import matplotlib.pyplot as plt
import numpy as np
import scipy


class UnscentedKalmanFilter(object):
    def __init__(self, x=np.zeros((4, 1)), P=np.eye(4)):
        self.x = x
        self.P = P
        # Covariance for EKF simulation
        self.Q = np.diag([
            0.1,  # variance of location on x-axis
            0.1,  # variance of location on y-axis
            np.deg2rad(1.0),  # variance of yaw angle
            1.0  # variance of velocity
        ]) ** 2  # predict state covariance
        self.R = np.diag([1.0, 1.0]) ** 2  # Observation x,y position covariance

        self.time = 0
        self.dt = 0.1

        self.dims = len(self.x)
        self.alpha = 1
        self.k = 2

        #  UKF Parameter
        self.ALPHA = 0.001
        self.BETA = 2
        self.KAPPA = 0
        self.gamma = None
        self.wm = None
        self.wc = None
        self.setup_ukf(self.dims)

    def setup_ukf(self, nx):
        lamb = self.ALPHA ** 2 * (nx + self.KAPPA) - nx
        # calculate weights
        wm = [lamb / (lamb + nx)]
        wc = [(lamb / (lamb + nx)) + (1 - self.ALPHA ** 2 + self.BETA)]
        for i in range(2 * nx):
            wm.append(1.0 / (2 * (nx + lamb)))
            wc.append(1.0 / (2 * (nx + lamb)))
        self.gamma = math.sqrt(nx + lamb)

        self.wm = np.array([wm])
        self.wc = np.array([wc])


    def returnSigma(self, mu, sigma):
        n = len(mu[:, 0])

        chi = mu
        for i in range(n):
            chi = np.hstack((chi, mu + self.gamma * scipy.linalg.sqrtm(sigma)[:, i:i + 1]))
        for i in range(n):
            chi = np.hstack((chi, mu - self.gamma * scipy.linalg.sqrtm(sigma)[:, i:i + 1]))
        return chi

    def motion_model(self, x, u):
        F = np.array([[1.0, 0, 0, 0],
                      [0, 1.0, 0, 0],
                      [0, 0, 1.0, 0],
                      [0, 0, 0, 0]])

        B = np.array([[self.dt * math.cos(x[2, 0]), 0],
                      [self.dt * math.sin(x[2, 0]), 0],
                      [0.0, self.dt],
                      [1.0, 0.0]])
        return F @ x + B @ u

    def observation_model(self, x):
        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        return H @ x

    def calculateGs(self, sigma, u):
        for i in range(sigma.shape[1]):
            sigma[:, i:i + 1] = self.motion_model(sigma[:, i:i + 1], u)
        return sigma

    def calculateHs(self, sigma):
        for i in range(sigma.shape[1]):
            sigma[0:2, i] = self.observation_model(sigma[:, i])

        return sigma[0:2, :]

    def calc_sigma_covariance(self, mu, sigma, Pi):
        nSigma = sigma.shape[1]
        d = sigma - mu[0:sigma.shape[0]]
        P = Pi
        for i in range(nSigma):
            P = P + self.wc[0, i] * d[:, i:i + 1] @ d[:, i:i + 1].T
        return P

    def calc_pxz(self, sigma, mu, z_sigma, zb):
        dx = sigma - mu
        dz = z_sigma - zb[0:2]
        P = np.zeros((dx.shape[0], dz.shape[0]))
        for i in range(sigma.shape[1]):
            P = P + self.wc[0, i] * dx[:, i:i + 1] @ dz[:, i:i + 1].T

        return P


    def filter(self, u, z):
        chi = self.returnSigma(self.x, self.P)
        chiStar = self.calculateGs(chi, u)
        mu = (self.wm @ chiStar.T).T

        sig = self.calc_sigma_covariance(mu, chiStar, self.Q)
        zPred = self.observation_model(mu)

        chiNew = self.returnSigma(mu, sig)
        zeta = self.calculateHs(chiNew)

        zHat = (self.wm @ zeta.T).T

        S = self.calc_sigma_covariance(zHat, zeta, self.R)
        sigXZ = self.calc_pxz(chiNew, mu, zeta, zHat)

        K = sigXZ @ np.linalg.inv(S)
        self.x = mu + K @ (z - zPred)
        self.P = sig - K @ S @ K.T
        print("-----")

        return self.x, self.P


def calc_input():
    v = 1.0  # [m/s]
    yawrate = 0.1  # [rad/s]
    u = np.array([[v], [yawrate]])
    return u


def main(xTrue=None):
    print(__file__ + " start!!")
    #  Simulation parameter
    INPUT_NOISE = np.diag([1.0, np.deg2rad(30.0)]) ** 2
    GPS_NOISE = np.diag([0.5, 0.5]) ** 2

    DT = 0.1  # time tick [s]
    SIM_TIME = 50.0  # simulation time [s]

    show_animation = True

    # State Vector [x y yaw v]'
    xEst = np.zeros((4, 1))
    xTrue = np.zeros((4, 1))
    PEst = np.eye(4)

    xDR = np.zeros((4, 1))  # Dead reckoning

    # history
    hxEst = xEst
    hxTrue = xTrue
    hxDR = xTrue
    hz = np.zeros((2, 1))

    filter = UnscentedKalmanFilter(xEst, PEst)

    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    def motion_model(x, u):
        F = np.array([[1.0, 0, 0, 0],
                      [0, 1.0, 0, 0],
                      [0, 0, 1.0, 0],
                      [0, 0, 0, 0]])

        B = np.array([[DT * math.cos(x[2, 0]), 0],
                      [DT * math.sin(x[2, 0]), 0],
                      [0.0, DT],
                      [1.0, 0.0]])

        x = F @ x + B @ u
        return x

    while SIM_TIME >= filter.time:
        u = calc_input()
        xTrue = motion_model(xTrue, u)

        z = H @ xTrue + GPS_NOISE @ np.random.randn(2, 1)
        ud = u + INPUT_NOISE @ np.random.randn(2, 1)
        xDR = motion_model(xDR, ud)

        xEst, pEst = filter.filter(ud, z)

        # store data history
        hxEst = np.hstack((hxEst, xEst))
        hxDR = np.hstack((hxDR, xDR))
        hxTrue = np.hstack((hxTrue, xTrue))
        hz = np.hstack((hz, z))

        if show_animation:
            plt.cla()
            # for stopping simulation with the esc key.
            plt.gcf().canvas.mpl_connect('key_release_event',
                    lambda event: [exit(0) if event.key == 'escape' else None])
            plt.plot(hz[0, :], hz[1, :], ".g")
            plt.plot(hxTrue[0, :].flatten(),
                     hxTrue[1, :].flatten(), "-b", label="True Path")
            plt.plot(hxDR[0, :].flatten(),
                     hxDR[1, :].flatten(), "-k", label="Dead Reckoning Path")
            plt.plot(hxEst[0, :].flatten(),
                     hxEst[1, :].flatten(), "-r", label="UKF Path")
            plt.legend()
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)


if __name__ == "__main__":
    main()
