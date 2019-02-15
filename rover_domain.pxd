# distutils: language = c++
# cython: language_level=3, boundscheck=True

from cython.view cimport array as cvarray
from cython.operator cimport address
from libcpp.vector cimport vector
from libcpp.functional cimport function
from libc cimport math as cmath
from libcpp.algorithm cimport partial_sort

cdef extern from "math.h":
    double sqrt(double m)

ctypedef enum ObjTypeId: ROVER_T_ID, POI_T_ID

cdef class RoverDomain:
    cdef public Py_ssize_t n_rovers
    cdef public Py_ssize_t n_pois
    cdef public Py_ssize_t n_steps
    cdef public Py_ssize_t n_req
    cdef public Py_ssize_t n_obs_sections
    cdef public double min_dist
    cdef public Py_ssize_t step_id
    cdef public bint done
    cdef public double setup_size
    cdef public double interaction_dist
    cdef public bint reorients
    cdef public bint discounts_poi_eval
    cdef public double[:, :] init_rover_positions
    cdef public double[:, :] init_rover_orientations
    cdef public double[:, :]rover_positions
    cdef public double[:, :, :] rover_position_histories
    cdef public double[:, :] rover_orientations
    cdef public double[:] poi_values
    cdef public double[:, :] poi_positions
    cdef public object update_rewards
    
    # Define module level temporary array manager for fast creation of short-lived
    # c++ arrays.
    # Note that the use of buffers make the a single instance of
    # the rover domain not thread-safe. Multiple instances can be on different
    # threads though
    cdef vector[double] sqr_dists_to_poi
    cdef vector[double] sqr_dists_to_poi_unsorted
    cdef vector[double] poi_evals
    cdef vector[double] actual_x_hist
    cdef vector[double] actual_y_hist
    
    # Some return values are stored for performance reasons
    cdef public double[:, :, :] rover_observations
    cdef public double[:] rover_rewards
    
    cpdef double[:, :, :] reset(self)
    cpdef void stop_prematurely(self)
    cpdef tuple step(self, double[:, :] actions)
    cpdef void step_no_ret(self, double[:, :] actions)
    cpdef double calc_step_eval_from_poi(self, Py_ssize_t poi_id)
    cpdef void update_local_step_reward_from_poi(self, Py_ssize_t poi_id)
    cpdef double calc_step_global_eval(self)
    cpdef double calc_step_cfact_global_eval(self, Py_ssize_t rover_id)
    cpdef double calc_traj_global_eval(self)
    cpdef double calc_traj_cfact_global_eval(self, Py_ssize_t rover_id)
    cpdef void add_to_sensor(self, Py_ssize_t rover_id, 
        ObjTypeId obj_type_id, double other_x, double other_y, double val)
    cpdef void update_observations(self)
    cpdef void update_rewards_step_global_eval(self)    
    cpdef void update_rewards_step_diff_eval(self)
    cpdef void update_rewards_traj_global_eval(self)    
    cpdef void update_rewards_traj_diff_eval(self)
        