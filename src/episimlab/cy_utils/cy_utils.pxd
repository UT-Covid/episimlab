cdef extern from "gsl/gsl_rng.h" nogil:
    ctypedef struct gsl_rng_type:
        pass
    ctypedef struct gsl_rng:
        pass
    gsl_rng_type *gsl_rng_mt19937
    gsl_rng *gsl_rng_alloc(gsl_rng_type * T)
    void gsl_rng_set(gsl_rng * r, unsigned long int)
    void gsl_rng_free(gsl_rng * r)

# Poisson distribution from GSL lib
cdef extern from "gsl/gsl_randist.h" nogil:
    unsigned int gsl_ran_poisson(gsl_rng * r, double mu)

cdef gsl_rng *get_seeded_rng(int int_seed) nogil

cdef double discrete_time_approx(double rate, double timestep) nogil
