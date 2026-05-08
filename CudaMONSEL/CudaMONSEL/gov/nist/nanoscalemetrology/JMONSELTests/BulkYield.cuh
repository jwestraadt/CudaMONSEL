// file: gov\nist\nanoscalemetrology\JMONSELTests\BulkYield.cuh

#ifndef _BULK_YIELD_CUH_
#define _BULK_YIELD_CUH_

#include "RuntimeInput.cuh"

namespace BulkYield
{
   void run();
   void run(const RuntimeInput::JsonValue& config);
}

#endif
