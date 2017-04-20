#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 30;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 30;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  n_aug_ = 7;

  // state dimension
  n_x_ = 5;

  //sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  //set vector for weights
  weights_ = VectorXd(2*n_aug_ + 1);

  P_ << 1,   0,    0,   0,   0,
        0,   1,    0,   0,   0,
        0,   0,    1,   0,   0,
        0,   0,    0,   1,   0,
        0,   0,    0,   0,   1;

   double weight_0 = lambda_ / (lambda_ + n_aug_);
   weights_(0) = weight_0;
   for (int i=1; i < 2 * n_aug_ + 1; i++) {  //2n+1 weights
	double weight = 0.5/(n_aug_ + lambda_);
	weights_(i) = weight;
   }

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  
  if (!is_initialized_) {
        if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
            float ro = meas_package.raw_measurements_[0];
            float phi = meas_package.raw_measurements_[1];
            float ro_dot = meas_package.raw_measurements_[2];
            float vx = ro_dot * cos(phi);
            float vy = ro_dot * sin(phi);
            x_ << ro * cos(phi),
            ro * sin(phi),
            0,
            0,
            0;
        }
        else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
            float x0 = meas_package.raw_measurements_[0];
            float x1 = meas_package.raw_measurements_[1];
            float px = (x0 == 0) ? 0.001 : x0;
            float py = (x1 == 0) ? 0.001 : x1;
            x_ << px,
            py,
            0,
            0,
            0;
        }

       
        is_initialized_ = true;
        time_us_ = meas_package.timestamp_;
        std::cout << "EKF initialized" << '\n';
        return;
   }



	double delta_t_  = (meas_package.timestamp_ - time_us_)  / 1000000.0;;
    

     
    Prediction(delta_t_);

    if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_) {
		std::cout<<"State  update radar\n";
        UpdateRadar(meas_package);
    } else if (meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
		std::cout<<"State  update lidar\n";
        UpdateLidar(meas_package);
    }
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  
	/////Generate Sigma Points
	 //create augmented mean vector
    VectorXd x_aug = VectorXd(n_aug_);

    //create augmented state covariance
    MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

    //create sigma point matrix
    MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);

    //create augmented mean state

    x_aug.head(n_x_) = x_;


    //create augmented covariance matrix
    P_aug.fill(0.0);
    P_aug.topLeftCorner(n_x_,n_x_) = P_;
    P_aug(5,5) = std_a_ * std_a_;
    P_aug(6,6) = std_yawdd_ * std_yawdd_;

    //create square root matrix
    MatrixXd L = P_aug.llt().matrixL();

    //create augmented sigma points
    Xsig_aug.col(0)  = x_aug;

    for (int i = 0; i< n_aug_; i++) {
        Xsig_aug.col(i+1)       = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
        Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
    }

    //predict sigma points
    for (int i = 0; i< 2*n_aug_+1; i++)  {
        //extract values for better readability
        double p_x = Xsig_aug(0,i);
        double p_y = Xsig_aug(1,i);
        double v = Xsig_aug(2,i);
        double yaw = Xsig_aug(3,i);
        double yawd = Xsig_aug(4,i);
        double nu_a = Xsig_aug(5,i);
        double nu_yawdd = Xsig_aug(6,i);

        //predicted state values
        double px_p, py_p;

        //avoid division by zero
        if (yawd !=0 ) {
            px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
            py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
        }
        else {
            px_p = p_x + v*delta_t*cos(yaw);
            py_p = p_y + v*delta_t*sin(yaw);
        }

        double v_p = v;
        double yaw_p = yaw + yawd*delta_t;
        double yawd_p = yawd;

        //add noise
        px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
        py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
        v_p = v_p + nu_a*delta_t;

        yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
        yawd_p = yawd_p + nu_yawdd*delta_t;

        //write predicted sigma point into right column
        Xsig_pred_(0,i) = px_p;
        Xsig_pred_(1,i) = py_p;
        Xsig_pred_(2,i) = v_p;
        Xsig_pred_(3,i) = yaw_p;
        Xsig_pred_(4,i) = yawd_p;
    }


	


	/////Predict Mean and Covariance
	 x_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
        x_ += weights_(i) * Xsig_pred_.col(i);
    }

    //predicted state covariance matrix
    P_.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        //angle normalization
        if (x_diff(3)> M_PI) x_diff(3) = remainder(x_diff(3), (2.*M_PI)) - M_PI;
        if (x_diff(3)<-M_PI) x_diff(3) = remainder(x_diff(3), (2.*M_PI)) + M_PI;

        P_ += weights_(i) * x_diff * x_diff.transpose();
    }

}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {

	int n_z = 2; 

	//create matrix for sigma points in measurement space
    MatrixXd Zsig = MatrixXd(n_z, n_z*n_aug_ + 1);
	VectorXd z_pred = VectorXd(2);
	MatrixXd S = MatrixXd(n_z,n_z);

	z_pred.fill(0.0);
	S.fill(0.0);
	
    for(unsigned int i = 0; i < 2*n_aug_ + 1; i++) {
        double px      = Xsig_pred_(0,i);
        double py      = Xsig_pred_(1,i);
        Zsig(0,i) = px;
        Zsig(1,i) = py;
		z_pred += (weights_(i) * Zsig.col(i));
    }

 
  

    //measurement covariance matrix S
    
    
    for (unsigned int i = 0; i < 2*n_aug_ + 1; i++) {
        VectorXd z_diff = Zsig.col(i) - z_pred;
        S += weights_(i) * z_diff * z_diff.transpose();
    }


    MatrixXd R(n_z,n_z);
    R << std_laspx_ * std_laspx_, 0,
    0, std_laspy_ * std_laspy_;

    S += R;

    //create vector for incoming lidar measurement
    VectorXd z = VectorXd(2);
    z <<meas_package.raw_measurements_(0),   //px
    meas_package.raw_measurements_(1);   //py

    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, n_z);
    Tc.fill(0.0);
    for (unsigned int j = 0; j < 2*n_aug_ + 1; j++) {
        VectorXd x_diff = Xsig_pred_.col(j) - x_;
        VectorXd z_diff = Zsig.col(j) - z_pred;
        // angle normalization
        if (x_diff(3)> M_PI) x_diff(3) = remainder(x_diff(3), (2.*M_PI)) - M_PI;
        if (x_diff(3)<-M_PI) x_diff(3) = remainder(x_diff(3), (2.*M_PI)) + M_PI;
        Tc += weights_(j) * x_diff * z_diff.transpose();
    }

    MatrixXd K = Tc * S.inverse();

    VectorXd z_diff = z - z_pred;

    x_ += (K * z_diff);
    P_ -= (K * S * K.transpose());
    NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
	//create matrix for sigma points in measurement space
	int n_z = 3; 


    MatrixXd Zsig = MatrixXd(n_z, 2 * n_aug_ + 1);
	VectorXd z_pred = VectorXd(n_z);
	MatrixXd S = MatrixXd(n_z, n_z);

	z_pred.fill(0.0);
	S.fill(0.0);


    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

        // extract values for better readibility
        double p_x = Xsig_pred_(0,i);
        double p_y = Xsig_pred_(1,i);
        double v  = Xsig_pred_(2,i);
        double yaw = Xsig_pred_(3,i);

        double v1 = cos(yaw)*v;
        double v2 = sin(yaw)*v;

        // measurement model
        Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
        Zsig(1,i) = atan2(p_y,p_x);                                 //phi
        Zsig(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);   //r_dot

		z_pred += weights_(i) * Zsig.col(i);
    }



    //measurement covariance matrix S
    
    
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points
        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;

        //angle normalization
        if (z_diff(1)> M_PI) z_diff(1) = remainder(z_diff(1), (2.*M_PI)) - M_PI;
        if (z_diff(1)<-M_PI) z_diff(1) = remainder(z_diff(1), (2.*M_PI)) + M_PI;

        S += weights_(i) * z_diff * z_diff.transpose();
    }

    //add measurement noise covariance matrix
    MatrixXd R = MatrixXd(n_z,n_z);
	R.fill(0);
    R <<    std_radr_ * std_radr_, 0, 0,
    0, std_radphi_ * std_radphi_, 0,
    0, 0,std_radrd_ * std_radrd_;
    S += R;


    //create matrix for cross correlation Tc
    MatrixXd Tc = MatrixXd(n_x_, n_z);

    //calculate cross correlation matrix
    Tc.fill(0.0);
    for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //2n+1 simga points

        //residual
        VectorXd z_diff = Zsig.col(i) - z_pred;
        //angle normalization
        if (z_diff(1)> M_PI) z_diff(1) = remainder(z_diff(1), (2.*M_PI)) - M_PI;
        if (z_diff(1)<-M_PI) z_diff(1) = remainder(z_diff(1), (2.*M_PI)) + M_PI;

        // state difference
        VectorXd x_diff = Xsig_pred_.col(i) - x_;
        //angle normalization
        if (x_diff(3)> M_PI) x_diff(3) = remainder(x_diff(3), (2.*M_PI)) - M_PI;
        if (x_diff(3)<-M_PI) x_diff(3) = remainder(x_diff(3), (2.*M_PI)) + M_PI;

        Tc += weights_(i) * x_diff * z_diff.transpose();
    }

    //Kalman gain K;
    MatrixXd K = Tc * S.inverse();

    //create vector for incoming radar measurement
    //---VectorXd z = VectorXd(3);
    //--z << meas_package.raw_measurements_[0],
    //--meas_package.raw_measurements_[1],
    //---meas_package.raw_measurements_[2];
	VectorXd z = meas_package.raw_measurements_;

    //residual
    VectorXd z_diff = z - z_pred;

    //angle normalization
    if (z_diff(1)> M_PI) z_diff(1) = remainder(z_diff(1), (2.*M_PI)) - M_PI;
    if (z_diff(1)<-M_PI) z_diff(1) = remainder(z_diff(1), (2.*M_PI)) + M_PI;

    //update state mean and covariance matrix
    x_ += K * z_diff;
    P_ -= K*S*K.transpose();
    NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
}
