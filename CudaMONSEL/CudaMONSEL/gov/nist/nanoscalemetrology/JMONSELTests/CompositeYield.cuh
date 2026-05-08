// file: gov\nist\nanoscalemetrology\JMONSELTests\CompositeYield.cuh

#ifndef _COMPOSITE_YIELD_CUH_
#define _COMPOSITE_YIELD_CUH_

#include "RuntimeInput.cuh"

namespace CompositeYield
{
   void run();
   void run(const RuntimeInput::JsonValue& config);
}

#endif
