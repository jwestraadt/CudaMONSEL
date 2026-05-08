// file: gov\nist\nanoscalemetrology\JMONSELTests\CompositeImage.cuh

#ifndef _COMPOSITE_IMAGE_CUH_
#define _COMPOSITE_IMAGE_CUH_

#include "RuntimeInput.cuh"

namespace CompositeImage
{
   void run();
   void run(const RuntimeInput::JsonValue& config);
}

#endif
