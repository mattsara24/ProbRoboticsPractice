import math

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation as Rot


class ExendedKalmanFilter(object):
    def __init__(self, x=np.zeros((4, 1)), P=np.eye(4)):
        self.mu = 0
        self.sigma = 0
        self.dt = 0.01

        self.J = np.array([[1.0, 0, 0, 0],
                      [0, 1.0, 0, 0]])

        # Covariance for EKF simulation
        self.Q = np.diag([
            0.1,  # variance of location on x-axis
            0.1,  # variance of location on y-axis
            np.deg2rad(1.0),  # variance of yaw angle
            1.0  # variance of velocity
        ]) ** 2  # predict state covariance
        self.R = np.diag([1.0, 1.0]) ** 2  # Observation x,y position covariance

        self.x = x
        self.P = P
        self.time = 0


    def motionModel(self, u):
        phi = self.x[2, 0]
        v = u[0, 0]

        F = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,0]])
        B = np.array([[np.cos(phi) * self.dt, 0], [np.sin(phi) * self.dt, 0], [0, self.dt], [1, 0]])

        J = np.array([
            [1.0, 0.0, -self.dt * v * np.sin(phi), self.dt * np.cos(phi)],
            [0.0, 1.0, self.dt * v * np.cos(phi), self.dt * np.sin(phi)],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]])

        return F, B, J

    def sensorModel(self):
        B = np.array([[1,0,0,0],[0,1,0,0]])
        J = np.array([[1,0,0,0],[0,1,0,0]])
        return B, J

    def predict(self, u):
        F, B, J = self.motionModel(u)

        self.x = F @ self.x + B @ u
        self.P = np.dot(J, np.dot(self.P, J.T)) + self.Q


    def update(self, z):
        B, J = self.sensorModel()
        zPred = B @ self.x
        y = z - zPred

        S = J @ self.P @ J.T + self.R
        K = self.P @ self.J.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(len(self.x)) - K @ J) @ self.P

    def filter(self, u, z):
        self.predict(u)
        self.update(z)
        self.time += self.dt
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

    filter = ExendedKalmanFilter(xEst, PEst)

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
                     hxEst[1, :].flatten(), "-r", label="EKF Path")
            plt.legend()
            plt.axis("equal")
            plt.grid(True)
            plt.pause(0.001)


if __name__ == "__main__":
    main()
