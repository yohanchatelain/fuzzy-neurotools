#include <dlfcn.h>
#include <math.h>

static double (*real_log)(double dbl);
static float (*real_logf)(float dbl);

// Override
double log(double dbl);
{
	real_log = dlsym(RTLD_NEXT, "log");
	return real_log(dbl) + 0.0;
}
float logf(float dbl);
{
        real_logf = dlsym(RTLD_NEXT, "logf");
        return real_logf(dbl) + 0.0f;
}
