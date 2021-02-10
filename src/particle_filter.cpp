/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;

// Random number engine generator for use across methods
static std::default_random_engine random_generator; 

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   *   Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. Add random Gaussian noise to each particle.
   */
  num_particles = 50;  // Set num of particles

  // Create normal distribution for initial position x, y, and theta for each particle
  std::normal_distribution<double> norm_dist_x(x, std[0]);
  std::normal_distribution<double> norm_dist_y(y, std[1]);
  std::normal_distribution<double> norm_dist_theta(theta, std[2]);
  
  for (auto i = 0; i < num_particles; i++)
  {
    Particle particle;                                  // Create particle
    particle.id = i;                                    // Assign particle ID
    particle.x = norm_dist_x(random_generator);         // Assign initial x position of particle
    particle.y = norm_dist_y(random_generator);         // Assign initial y position of particle
    particle.theta = norm_dist_theta(random_generator); // Set initial theta position of particle
    particle.weight = 1.0;                              // Initialize particle weight to 1

    // Add particle to set of current particles
    this->particles.push_back(particle);

    // Initialize weights vector to 1.0 for all particles
    this->weights.push_back(1.0);
  }

  // Show particles for debugging
  // this->show_particles(particles);

  // Particle filter initialized flag
  this->is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
  * Add measurements to each particle and add random Gaussian noise.
  */

  // Creat normal distribution for measurement noise
  std::normal_distribution<double> norm_dist_x(0, std_pos[0]);
  std::normal_distribution<double> norm_dist_y(0, std_pos[1]);
  std::normal_distribution<double> norm_dist_theta(0, std_pos[2]);

  // Loop through all particles and update position
  for (int i = 0; i < this->num_particles; i++)
  {
    // Get theta for current particle
    double theta = this->particles[i].theta;

    // Case where yaw rate is very close to 0
    if (fabs(yaw_rate) < 0.0001)
    {
      // Update particles pose
      this->particles[i].x += velocity * delta_t * cos(theta);
      this->particles[i].x += velocity * delta_t * sin(theta);
      // Yaw rate isn't changing so theta is the same
    }
    // Yaw rate is changing 
    else
    {
      // Update particles pose
      this->particles[i].x += velocity / yaw_rate * (sin(theta + yaw_rate * delta_t) - sin(theta));
      this->particles[i].y += velocity / yaw_rate * (cos(theta) - cos(theta + yaw_rate * delta_t));
      this->particles[i].theta += delta_t * yaw_rate;
    }

    // Add measurement noise to each particle
    this->particles[i].x += norm_dist_x(random_generator);
    this->particles[i].y += norm_dist_y(random_generator);
    this->particles[i].theta += norm_dist_theta(random_generator);
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   *   Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   */

  // Loop through observations
  for (unsigned int i = 0; i < observations.size(); i++)
  { 
    // Current observation
    LandmarkObs obs = observations[i];

    // Set minimum distance for a data association to be large number initially
    double min_distance = std::numeric_limits<double>::max();

    // Initialize landmark id  
    int landmark_id = -1;

    // landmark index
    int idx = -1;

    // Loop through predicitons 
    for (unsigned int j = 0; j < predicted.size(); j++)
    {
      // Current prediction
      LandmarkObs prd = predicted[j];

      // Compute distance between observation and prediction
      double distance = dist(obs.x, obs.y, prd.x, prd.y);

      // The landmark closest to the observation will be associated
      if (distance < min_distance)
      {
        min_distance = distance;
        landmark_id = prd.id;
        idx = prd.idx;
      }
    }
    // Associate observation to closest landmark
    observations[i].id = landmark_id;
    observations[i].idx = idx;
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   *   Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   */

  // Uncertainty in landmark mesaurements
  double std_x = std_landmark[0];
  double std_y = std_landmark[1];

  // Loop through each particle
  for (int i = 0; i < num_particles; i++)
  {
    // Used for debugging visualization
    std::vector<int> associations;
    std::vector<double> sense_x;
    std::vector<double> sense_y;

    // Get pose of particle
    double p_x = this->particles[i].x;
    double p_y = this->particles[i].y;
    double p_theta = this->particles[i].theta;

    // Vector of LandMarkObs to store landmarks that are in sensor range
    std::vector<LandmarkObs> predicted_landmarks;

    // Loop through each landmark in the map and determine if landmark
    // is within sensor range of particle
    for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++)
    {
      // Get x and y coordinate and landmark id
      double landmark_x = map_landmarks.landmark_list[j].x_f;
      double landmark_y = map_landmarks.landmark_list[j].y_f;
      int landmark_id = map_landmarks.landmark_list[j].id_i;

      // Compute distance between landmark and particle
      if (dist(p_x, p_y, landmark_x, landmark_y) <= sensor_range)
      {
        predicted_landmarks.push_back(LandmarkObs{landmark_id, landmark_x, landmark_y, (int)j});
      }
    }

    // Transform observations from sensor from vehicle coordinates to map coordinates
    std::vector<LandmarkObs> t_observations;

    for (unsigned int j = 0; j < observations.size(); j++)
    {
      double t_x = cos(p_theta) * observations[j].x - sin(p_theta) * observations[j].y + p_x;
      double t_y = sin(p_theta) * observations[j].x + cos(p_theta) * observations[j].y + p_y;
      t_observations.push_back(LandmarkObs{observations[j].id, t_x, t_y});
    }

    // Associate landmarks with observations 
    dataAssociation(predicted_landmarks, t_observations);

    // Reset current particles weight before update
    particles[i].weight = 1.0;

    // Loop through transformed observations
    for (unsigned int j = 0; j < t_observations.size(); j++)
    {
      // Get current observation
      LandmarkObs obs = t_observations[j];

      // Get the landmark id associated with current observation
      int associated_lndmrk_id = t_observations[j].id;
      int lndmark_idx = t_observations[j].idx;

      double lm_x = map_landmarks.landmark_list[lndmark_idx].x_f;
      double lm_y = map_landmarks.landmark_list[lndmark_idx].y_f;

      // @INFO - Eliminated this for loop by adding index variable to
      // LandmarkObs struct so I can idnex directly into map_landmarks.landmark_list

      // Get coordinates of landmark associated with current observation
      // for (auto k = 0; k < predicted_landmarks.size(); k++)
      // {
      //   if (predicted_landmarks[k].id == associated_lndmrk_id)
      //   {
      //     lm_x = predicted_landmarks[k].x;
      //     lm_y = predicted_landmarks[k].y;
      //   }
      // }
 
      // Calculate weight
      double weight =  1 / (2*M_PI*std_x*std_y) * exp(-0.5*(pow((obs.x-lm_x)/std_x,2) + pow((obs.y-lm_y)/std_y,2)));

      // Update weight
      this->particles[i].weight *= weight;

      associations.push_back(associated_lndmrk_id);
      sense_x.push_back(obs.x);
      sense_y.push_back(obs.y);
      this->weights[i] = this->particles[i].weight;
      SetAssociations(particles[i], associations, sense_x, sense_y);
    }
  }
}

void ParticleFilter::resample() {
  /**
   *   Resample particles with replacement with probability proportional 
   *   to their weight. 
   */

  // Vector to store resampled particles
  std::vector<Particle> resampled_particles;

  // Create unifrom distribution to pick index from
  std::uniform_int_distribution<int> start_index(0, num_particles - 1);

  // Get random starting index
  auto curr_index = start_index(random_generator);

  double beta = 0.0;

  double max_weight = 2.0 * *std::max_element(this->weights.begin(), this->weights.end());

  for (unsigned int i = 0; i < particles.size(); i++)
  {
    std::uniform_real_distribution<double> rand_weight(0.0, max_weight);
    beta += rand_weight(random_generator);

    while (beta > weights[curr_index])
    {
      beta -= this->weights[curr_index];
      curr_index = (curr_index + 1) % this->num_particles;
    }
    resampled_particles.push_back(this->particles[curr_index]);
  }
    this->particles = resampled_particles;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations = associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

void ParticleFilter::show_particles(const std::vector<Particle>& p)
{
  //Loop through set of particles and print x, y, theta, and weight
  for (auto i = 0; i < int(p.size()); ++i)
  {
    std::cout << "X: " << p[i].x << std::endl;
    std::cout << "Y: " << p[i].y << std::endl;
    std::cout << "Theta: " << p[i].theta << std::endl;
  }
} 