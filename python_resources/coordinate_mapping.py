import numpy as np


qcar_pose=np.array([[-1.20,-0.83],[0.97,-1.08],[2.21,-0.15],[2.24,3.52],[0.38,4.40],[-0.41,4.27],[-1.84,2.39],[-0.92,0.83]])
ros_pose=np.array([[-1.20,-0.66],[0.23,-1.18],[2.12,-0.56],[2.72,2.92],[1.36,4.25],[0.26,4.28],[-1.42,3.24],[-0.78,0.97]])

X = qcar_pose
Y = ros_pose

# Add ones column to solve for translation too
X_aug = np.hstack([X, np.ones((X.shape[0], 1))])

# Solve least squares
params, _, _, _ = np.linalg.lstsq(X_aug, Y, rcond=None)

A = params[:2, :]   # 2x2 matrix
b = params[2, :]    # translation vector

print("A =\n", A)
print("b =\n", b)