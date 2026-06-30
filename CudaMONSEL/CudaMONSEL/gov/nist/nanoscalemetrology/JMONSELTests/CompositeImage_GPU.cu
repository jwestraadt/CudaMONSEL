#include "gov\nist\nanoscalemetrology\JMONSELTests\CompositeImage_GPU.cuh"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <vector>

#define CM_GPU_PI 3.14159265358979323846
#define CM_GPU_ELECTRON_CHARGE 1.60217653e-19
#define CM_GPU_BOHR_RADIUS 5.291772083e-11
#define CM_GPU_CHAMBER_RADIUS 0.1
#define CM_GPU_SMALL_DISP 1.0e-15
#define CM_GPU_MAX_NISTMOTT (20000.0 * CM_GPU_ELECTRON_CHARGE)
#define CM_GPU_DL50 -39.36405396577959
#define CM_GPU_NIST_PARAM 0.09986021173361583

namespace CompositeImageGPU
{
   namespace
   {
      static const int REGION_NONE = -1;
      static const int REGION_VAC = 0;
      static const int REGION_SL = 1;
      static const int REGION_BULK = 2;
      static const int REGION_PRECIP = 3;
      static const int TYPE_PRIMARY = 0;
      static const int TYPE_SE1 = 1;
      static const int TYPE_SE2 = 2;
      static const int MAX_SECONDARY_STACK = 64;
      static const int MAX_STEPS_PER_ELECTRON = 200000;
      static const int LAUNCH_BATCH_TRAJ = 65536;

      struct Vec3
      {
         double x, y, z;
      };

      struct ElectronState
      {
         Vec3 pos;
         double theta;
         double phi;
         double energy;
         double previousEnergy;
         int region;
         int type;
         int steps;
         bool complete;
      };

      struct BoundaryHit
      {
         double t;
         int nextRegion;
         Vec3 normalTowardNext;
         bool hit;
      };

      struct KernelConfig
      {
         const MatGPU* mats;
         const ElemTableGPU* elems;
         const double* pixelX;
         const double* pixelY;
         int* seCounts;
         int* se1Counts;
         int* se2Counts;
         int* totalCounts;
         int* genSECounts;
         int pixelCount;
         int trajPerPixel;
         int startTrajectory;
         int batchTrajectoryCount;
         GeomGPU geom;
         double beamE;
         double beamSizeM;
         double beamStartZ;
         double seThresholdJ;
         unsigned long long seed;
         int*   escapeHist;        // per-pixel [energy_bin][angle_bin], nullptr if disabled
         int    histNEbins;
         int    histNBbins;
         double histEbinWidthJ;
      };

      __host__ __device__ double sqr(double x)
      {
         return x * x;
      }

      __device__ Vec3 add(Vec3 a, Vec3 b)
      {
         return { a.x + b.x, a.y + b.y, a.z + b.z };
      }

      __device__ Vec3 sub(Vec3 a, Vec3 b)
      {
         return { a.x - b.x, a.y - b.y, a.z - b.z };
      }

      __device__ Vec3 mul(double s, Vec3 a)
      {
         return { s * a.x, s * a.y, s * a.z };
      }

      __device__ double dot(Vec3 a, Vec3 b)
      {
         return a.x * b.x + a.y * b.y + a.z * b.z;
      }

      __device__ double norm2(Vec3 a)
      {
         return dot(a, a);
      }

      __device__ Vec3 normalize(Vec3 a)
      {
         double n = sqrt(norm2(a));
         if (n <= 0.0) return { 0.0, 0.0, 1.0 };
         return mul(1.0 / n, a);
      }

      __device__ double clampUnit(double x)
      {
         if (x < -1.0) return -1.0;
         if (x > 1.0) return 1.0;
         return x;
      }

      struct Rng
      {
         unsigned long long state;

         __device__ unsigned long long next()
         {
            unsigned long long z = (state += 0x9e3779b97f4a7c15ULL);
            z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
            z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
            return z ^ (z >> 31);
         }

         __device__ double uniform()
         {
            return (next() >> 11) * (1.0 / 9007199254740992.0);
         }

         __device__ double openUniform()
         {
            double u = uniform();
            return u <= 0.0 ? 1.0 / 9007199254740992.0 : u;
         }
      };

      __device__ int regionAt(Vec3 p, const GeomGPU& geom)
      {
         if (norm2(p) > CM_GPU_CHAMBER_RADIUS * CM_GPU_CHAMBER_RADIUS)
            return REGION_NONE;

         if (geom.hasSL) {
            if (p.z >= 0.0) {
               double dx = p.x - geom.precipCx;
               double dy = p.y - geom.precipCy;
               double dz = p.z - geom.precipCz;
               return (dx * dx + dy * dy + dz * dz <= geom.precipR2) ? REGION_PRECIP : REGION_BULK;
            }
            return (p.z >= -geom.slThick) ? REGION_SL : REGION_VAC;
         }

         if (p.z >= 0.0) {
            double dx = p.x - geom.precipCx;
            double dy = p.y - geom.precipCy;
            double dz = p.z - geom.precipCz;
            return (dx * dx + dy * dy + dz * dz <= geom.precipR2) ? REGION_PRECIP : REGION_BULK;
         }
         return REGION_VAC;
      }

      __device__ double planeT(Vec3 p0, Vec3 p1, double zPlane)
      {
         double dz = p1.z - p0.z;
         if (dz == 0.0) return INFINITY;
         double t = (zPlane - p0.z) / dz;
         return (t > 0.0 && t <= 1.0) ? t : INFINITY;
      }

      __device__ double sphereT(Vec3 p0, Vec3 p1, const GeomGPU& geom)
      {
         Vec3 c = { geom.precipCx, geom.precipCy, geom.precipCz };
         Vec3 d = sub(p1, p0);
         Vec3 m = sub(p0, c);
         double a = dot(d, d);
         if (a == 0.0) return INFINITY;
         double b = 2.0 * dot(m, d);
         double cc = dot(m, m) - geom.precipR2;
         double disc = b * b - 4.0 * a * cc;
         if (disc < 0.0) return INFINITY;
         double root = sqrt(disc);
         double t0 = (-b - root) / (2.0 * a);
         double t1 = (-b + root) / (2.0 * a);
         double res = INFINITY;
         if (t0 > 0.0 && t0 <= 1.0) res = t0;
         if (t1 > 0.0 && t1 <= 1.0 && t1 < res) res = t1;
         return res;
      }

      __device__ double chamberT(Vec3 p0, Vec3 p1)
      {
         Vec3 d = sub(p1, p0);
         double a = dot(d, d);
         if (a == 0.0) return INFINITY;
         double b = 2.0 * dot(p0, d);
         double cc = dot(p0, p0) - CM_GPU_CHAMBER_RADIUS * CM_GPU_CHAMBER_RADIUS;
         double disc = b * b - 4.0 * a * cc;
         if (disc < 0.0) return INFINITY;
         double root = sqrt(disc);
         double t0 = (-b - root) / (2.0 * a);
         double t1 = (-b + root) / (2.0 * a);
         double res = INFINITY;
         if (t0 > 0.0 && t0 <= 1.0) res = t0;
         if (t1 > 0.0 && t1 <= 1.0 && t1 < res) res = t1;
         return res;
      }

      __device__ void considerHit(BoundaryHit& best, double t, int nextRegion, Vec3 normal)
      {
         if (t > 0.0 && t <= 1.0 && t < best.t) {
            best.t = t;
            best.nextRegion = nextRegion;
            best.normalTowardNext = normal;
            best.hit = true;
         }
      }

      __device__ int regionJustPast(Vec3 p0, Vec3 p1, double t, Vec3 dir, const GeomGPU& geom)
      {
         Vec3 delta = sub(p1, p0);
         Vec3 hit = add(p0, mul(t, delta));
         return regionAt(add(hit, mul(CM_GPU_SMALL_DISP, dir)), geom);
      }

      __device__ BoundaryHit findBoundary(int region, Vec3 p0, Vec3 p1, const GeomGPU& geom, Vec3 dir)
      {
         BoundaryHit best;
         best.t = INFINITY;
         best.nextRegion = region;
         best.normalTowardNext = dir;
         best.hit = false;

         if (region == REGION_VAC) {
            if (geom.hasSL) {
               double t = planeT(p0, p1, -geom.slThick);
               considerHit(best, t, regionJustPast(p0, p1, t, dir, geom), { 0.0, 0.0, 1.0 });
            }
            else {
               double t = planeT(p0, p1, 0.0);
               considerHit(best, t, regionJustPast(p0, p1, t, dir, geom), { 0.0, 0.0, 1.0 });
            }
            considerHit(best, chamberT(p0, p1), REGION_NONE, dir);
         }
         else if (region == REGION_SL) {
            considerHit(best, planeT(p0, p1, -geom.slThick), REGION_VAC, { 0.0, 0.0, -1.0 });
            double t = planeT(p0, p1, 0.0);
            considerHit(best, t, regionJustPast(p0, p1, t, dir, geom), { 0.0, 0.0, 1.0 });
         }
         else if (region == REGION_BULK) {
            considerHit(best, planeT(p0, p1, 0.0), geom.hasSL ? REGION_SL : REGION_VAC, { 0.0, 0.0, -1.0 });
            double ts = sphereT(p0, p1, geom);
            if (ts < INFINITY)
               considerHit(best, ts, REGION_PRECIP, dir);
            considerHit(best, chamberT(p0, p1), REGION_NONE, dir);
         }
         else if (region == REGION_PRECIP) {
            considerHit(best, sphereT(p0, p1, geom), REGION_BULK, dir);
            // Use dir (electron direction) as the z=0 surface normal to match the CPU
            // ExpQMBarrierSM fallback: precipSphere is not a NormalShape so CPU uses
            // nb=n0 (dir), giving cosalpha=1 and unconditional transmission when kE>deltaU.
            considerHit(best, planeT(p0, p1, 0.0), geom.hasSL ? REGION_SL : REGION_VAC, dir);
            considerHit(best, chamberT(p0, p1), REGION_NONE, dir);
         }

         return best;
      }

      __device__ double uNeville(const double* f, int offset, int order, double x)
      {
         double c[4];
         double d[4];
         for (int i = 0; i <= order; ++i) {
            c[i] = f[offset + i];
            d[i] = f[offset + i];
         }

         int ns = (int)floor(x + 0.5);
         if (ns < 0) ns = 0;
         if (ns > order) ns = order;

         double y = c[ns--];
         for (int m = 1; m <= order; ++m) {
            for (int i = 0; i <= order - m; ++i) {
               double ho = i - x;
               double hp = i + m - x;
               double w = c[i + 1] - d[i];
               d[i] = -hp * w / m;
               c[i] = -ho * w / m;
            }
            y += (2 * ns < (order - 1 - m)) ? c[ns + 1] : d[ns--];
         }
         return y;
      }

      __device__ double lagrangeD1(const double* f, int len, double x0, double xinc, int order, double x)
      {
         double reducedx = (x - x0) / xinc;
         int index0 = (int)reducedx - order / 2;
         if (index0 < 0) index0 = 0;
         if (index0 > len - order - 1) index0 = len - order - 1;
         return uNeville(f, index0, order, reducedx - index0);
      }

      __device__ double lagrangeD2(const double* f, int rows, int cols, const double x0[2], const double xinc[2], int order, const double x[2])
      {
         double reducedx1 = (x[0] - x0[0]) / xinc[0];
         int index0 = (int)reducedx1 - order / 2;
         if (index0 < 0) index0 = 0;
         if (index0 > rows - order - 1) index0 = rows - order - 1;

         double y[4];
         for (int i = 0; i <= order; ++i)
            y[i] = lagrangeD1(f + (index0 + i) * cols, cols, x0[1], xinc[1], order, x[1]);
         return lagrangeD1(y, order + 1, x0[0] + index0 * xinc[0], xinc[0], order, x[0]);
      }

      __device__ double screenedRutherfordTotal(const ElemTableGPU& elem, double energy)
      {
         double z13 = pow((double)elem.Z, 1.0 / 3.0);
         return (7.670843088080456e-38 * z13 * (1.0 + elem.Z)) /
            (energy + 5.44967975966321e-19 * z13 * z13);
      }

      __device__ double screenedRutherfordAngle(const ElemTableGPU& elem, double energy, Rng& rng)
      {
         double alpha = (5.44968e-19 * pow((double)elem.Z, 2.0 / 3.0)) / energy;
         double r = rng.openUniform();
         return acos(clampUnit(1.0 - 2.0 * alpha * r / (1.0 + alpha - r)));
      }

      __device__ double browningTotal(const ElemTableGPU& elem, double energy)
      {
         double eKeV = energy / (1000.0 * CM_GPU_ELECTRON_CHARGE);
         double re = sqrt(eKeV);
         return 3.0e-22 * elem.Zp17 /
            (eKeV + 0.005 * elem.Zp17 * re + 0.0007 * elem.Zp2 / re);
      }

      __device__ double browningAngle(const ElemTableGPU& elem, double energy, Rng& rng)
      {
         double r1 = rng.openUniform();
         double r2 = rng.openUniform();
         double eKeV = energy / (1000.0 * CM_GPU_ELECTRON_CHARGE);
         double r = (300.0 * eKeV / elem.Z) + (elem.Zp3 / (3.0e5 * eKeV));
         if (r1 <= r / (r + 1.0)) {
            double alpha = 7.0e-3 / eKeV;
            return acos(clampUnit(1.0 - ((2.0 * alpha * r2) / (alpha - r2 + 1.0))));
         }
         return acos(clampUnit(1.0 - 2.0 * r2));
      }

      __device__ double nistTotal(const ElemTableGPU& elem, double energy)
      {
         if (energy < elem.extraBelowE)
            return elem.sfBrowning * browningTotal(elem, energy);
         if (energy < CM_GPU_MAX_NISTMOTT)
            return CM_GPU_BOHR_RADIUS * CM_GPU_BOHR_RADIUS *
               lagrangeD1(elem.spwem, SPWEM_LEN, CM_GPU_DL50, CM_GPU_NIST_PARAM, 3, log(energy));
         return screenedRutherfordTotal(elem, energy);
      }

      __device__ double nistAngle(const ElemTableGPU& elem, double energy, Rng& rng)
      {
         if (energy < elem.extraBelowE)
            return browningAngle(elem, energy, rng);
         if (energy < CM_GPU_MAX_NISTMOTT) {
            double x0[2] = { CM_GPU_DL50, 0.0 };
            double xinc[2] = { CM_GPU_NIST_PARAM, 0.005 };
            double x[2] = { log(energy), rng.openUniform() };
            double q = lagrangeD2(elem.x1, SPWEM_LEN, X1_LEN, x0, xinc, 3, x);
            return acos(clampUnit(1.0 - 2.0 * q * q));
         }
         return screenedRutherfordAngle(elem, energy, rng);
      }

      __device__ double csdLossPositive(const MatGPU& mat, double len, double kE)
      {
         double loss = 0.0;
         for (int i = 0; i < mat.nCSD; ++i)
            loss += mat.coefJL[i] * log((mat.recipJ[i] * kE) + mat.betaJL[i]);
         return loss * len / kE;
      }

      __device__ double csdCompute(const MatGPU& mat, double len, double kE)
      {
         if (mat.nCSD == 0 || kE < mat.minEtrack || kE <= 0.0)
            return 0.0;
         if (kE <= mat.breakE)
            return (kE / pow(1.0 + (1.5 * mat.gammaN * len * kE * sqrt(kE)), 2.0 / 3.0)) - kE;

         double firstLoss = csdLossPositive(mat, len, kE);
         if (firstLoss <= 0.1 * kE)
            return -firstLoss;

         int pieces = 2;
         while (pieces < 64 && csdLossPositive(mat, len / pieces, kE) > 0.1 * kE)
            pieces *= 2;

         double energy = kE;
         double totalLoss = 0.0;
         double chunk = len / pieces;
         for (int i = 0; i < pieces; ++i) {
            double loss = csdLossPositive(mat, chunk, energy);
            if (loss > 0.5 * energy)
               loss = 0.5 * energy;
            energy -= loss;
            totalLoss += loss;
            if (energy <= mat.minEtrack || energy <= 0.0)
               break;
         }
         return -totalLoss;
      }

      __device__ Vec3 directionFromAngles(double theta, double phi)
      {
         double st = sin(theta);
         return { cos(phi) * st, sin(phi) * st, cos(theta) };
      }

      __device__ void updateDirection(ElectronState& e, double dTheta, double dPhi)
      {
         double ct = cos(e.theta), st = sin(e.theta);
         double cp = cos(e.phi), sp = sin(e.phi);
         double ca = cos(dTheta), sa = sin(dTheta);
         double cb = cos(dPhi);

         double xx = cb * ct * sa + ca * st;
         double yy = sa * sin(dPhi);
         double dx = cp * xx - sp * yy;
         double dy = cp * yy + sp * xx;
         double dz = ca * ct - cb * sa * st;

         e.theta = atan2(sqrt(dx * dx + dy * dy), dz);
         e.phi = atan2(dy, dx);
      }

      __device__ void ratesForMaterial(
         const MatGPU& mat,
         const ElemTableGPU* elems,
         double energy,
         double& elasticRate,
         double& inelasticRate,
         double cumulativeElastic[MAX_MAT_ELEM])
      {
         elasticRate = 0.0;
         inelasticRate = 0.0;

         if (mat.isVacuum)
            return;

         for (int i = 0; i < mat.nElems; ++i) {
            elasticRate += nistTotal(elems[mat.elemIdx[i]], energy) * mat.scalefactor[i];
            cumulativeElastic[i] = elasticRate;
         }
         elasticRate *= mat.densityNa;

         if (energy > (mat.energySEgen + mat.eFermi))
            inelasticRate = (-csdCompute(mat, 1.0e-10, energy) * 1.0e10) / mat.energySEgen;
      }

      __device__ void scatterElastic(
         ElectronState& e,
         const MatGPU& mat,
         const ElemTableGPU* elems,
         const double cumulativeElastic[MAX_MAT_ELEM],
         double elasticRate,
         Rng& rng)
      {
         if (elasticRate <= 0.0)
            return;

         double scaledTotal = elasticRate / mat.densityNa;
         double r = rng.openUniform() * scaledTotal;
         int index = 0;
         while (index + 1 < mat.nElems && cumulativeElastic[index] < r)
            ++index;

         double alpha = nistAngle(elems[mat.elemIdx[index]], e.previousEnergy, rng);
         double beta = 2.0 * CM_GPU_PI * rng.openUniform();
         updateDirection(e, alpha, beta);
      }

      __device__ bool scatterInelastic(
         ElectronState& e,
         const MatGPU& mat,
         ElectronState& secondary,
         Rng& rng)
      {
         secondary = e;
         secondary.theta = acos(clampUnit(1.0 - (2.0 * rng.openUniform())));
         secondary.phi = 2.0 * CM_GPU_PI * rng.openUniform();
         secondary.energy = mat.energySEgen + mat.eFermi;
         secondary.previousEnergy = secondary.energy;
         secondary.steps = 0;
         secondary.complete = false;
         return secondary.energy > mat.minEtrack;
      }

      __device__ bool applyBarrier(
         ElectronState& e,
         const MatGPU* mats,
         int nextRegion,
         Vec3 normalTowardNext)
      {
         if (nextRegion == REGION_NONE) {
            e.region = REGION_NONE;
            e.complete = true;
            return true;
         }

         const MatGPU& currentMat = mats[e.region];
         const MatGPU& nextMat = mats[nextRegion];
         double deltaU = currentMat.isVacuum ? 0.0 : -currentMat.energyCBbottom;
         if (!nextMat.isVacuum)
            deltaU += nextMat.energyCBbottom;

         Vec3 nb = normalize(normalTowardNext);
         Vec3 n0 = directionFromAngles(e.theta, e.phi);
         double cosalpha = dot(n0, nb);

         if (cosalpha <= 0.0) {
            e.pos = sub(e.pos, mul(CM_GPU_SMALL_DISP, nb));
            return false;
         }

         Vec3 nf;
         bool transmits = false;
         double kE0 = e.energy;
         double perpE = kE0 <= 0.0 ? 0.0 : cosalpha * cosalpha * kE0;
         double rootPerpE = 0.0;
         double rootDiff = 0.0;

         if (deltaU == 0.0) {
            transmits = true;
         }
         else if ((perpE != 0.0) && (perpE > deltaU)) {
            rootPerpE = sqrt(perpE);
            rootDiff = sqrt(perpE - deltaU);
            transmits = true;
         }

         if (transmits) {
            if (deltaU == 0.0) {
               nf = n0;
            }
            else {
               double factor = cosalpha * ((rootDiff / rootPerpE) - 1.0);
               nf = add(n0, mul(factor, nb));
               nf = normalize(nf);
            }
            e.energy = kE0 - deltaU;
            e.region = nextRegion;
            e.pos = add(e.pos, mul(CM_GPU_SMALL_DISP, nb));
         }
         else {
            nf = sub(n0, mul(2.0 * cosalpha, nb));
            nf = normalize(nf);
            e.pos = sub(e.pos, mul(CM_GPU_SMALL_DISP, nb));
         }

         e.theta = acos(clampUnit(nf.z));
         e.phi = atan2(nf.y, nf.x);
         return transmits;
      }

      __device__ void recordEscape(const ElectronState& e, const KernelConfig& cfg, int pixel)
      {
         atomicAdd(&cfg.totalCounts[pixel], 1);
         if (e.energy < cfg.seThresholdJ) {
            atomicAdd(&cfg.seCounts[pixel], 1);
            if (e.type == TYPE_SE1)
               atomicAdd(&cfg.se1Counts[pixel], 1);
            else if (e.type == TYPE_SE2)
               atomicAdd(&cfg.se2Counts[pixel], 1);
         }

         // Optional (escape energy x take-off angle) histogram. beta is the
         // take-off polar angle from the outward optic axis (-z): beta = pi - theta,
         // so beta = 0 is straight up the column, beta = 90 deg is grazing.
         if (cfg.escapeHist != nullptr) {
            int ie = (int)(e.energy / cfg.histEbinWidthJ);
            if (ie < 0) ie = 0;
            if (ie >= cfg.histNEbins) ie = cfg.histNEbins - 1;
            double beta = CM_GPU_PI - e.theta;
            double bwidth = (CM_GPU_PI * 0.5) / cfg.histNBbins;
            int ib = (int)(beta / bwidth);
            if (ib < 0) ib = 0;
            if (ib >= cfg.histNBbins) ib = cfg.histNBbins - 1;
            atomicAdd(&cfg.escapeHist[(pixel * cfg.histNEbins + ie) * cfg.histNBbins + ib], 1);
         }
      }

      __device__ void initializePrimary(ElectronState& e, const KernelConfig& cfg, int pixel, Rng& rng)
      {
         double r = sqrt(-2.0 * log(rng.openUniform())) * cfg.beamSizeM;
         double th = 2.0 * CM_GPU_PI * rng.openUniform();
         e.pos = {
            cfg.pixelX[pixel] + r * cos(th),
            cfg.pixelY[pixel] + r * sin(th),
            cfg.beamStartZ
         };
         e.theta = 0.0;
         e.phi = 0.0;
         e.energy = cfg.beamE;
         e.previousEnergy = cfg.beamE;
         e.region = regionAt(e.pos, cfg.geom);
         e.type = TYPE_PRIMARY;
         e.steps = 0;
         e.complete = false;
      }

      __device__ void runOneTrajectory(const KernelConfig& cfg, int pixel, unsigned long long trajectoryId)
      {
         Rng rng;
         rng.state = cfg.seed ^ (trajectoryId * 0xd1b54a32d192ed03ULL);

         ElectronState e;
         initializePrimary(e, cfg, pixel, rng);

         ElectronState stack[MAX_SECONDARY_STACK];
         int stackTop = 0;

         while (true) {
            while (!e.complete) {
               e.region = regionAt(e.pos, cfg.geom);
               if (e.region == REGION_NONE) {
                  e.complete = true;
                  break;
               }

               const MatGPU& mat = cfg.mats[e.region];
               if (e.energy < mat.minEtrack || e.energy <= 0.0 || e.steps > MAX_STEPS_PER_ELECTRON) {
                  e.complete = true;
                  break;
               }

               double cumulativeElastic[MAX_MAT_ELEM];
               double elasticRate = 0.0;
               double inelasticRate = 0.0;
               ratesForMaterial(mat, cfg.elems, e.energy, elasticRate, inelasticRate, cumulativeElastic);
               double totalRate = elasticRate + inelasticRate;
               double freePath = (totalRate > 0.0) ? (-log(rng.openUniform()) / totalRate) : (2.0 * CM_GPU_CHAMBER_RADIUS);

               Vec3 dir = directionFromAngles(e.theta, e.phi);
               Vec3 candidate = add(e.pos, mul(freePath, dir));
               BoundaryHit hit = findBoundary(e.region, e.pos, candidate, cfg.geom, dir);
               double stepLen = freePath;
               if (hit.hit) {
                  stepLen *= hit.t;
                  candidate = add(e.pos, mul(stepLen, dir));
               }

               e.previousEnergy = e.energy;
               e.pos = candidate;
               e.energy += csdCompute(mat, stepLen, e.energy);
               ++e.steps;

               if (e.energy < mat.minEtrack || e.energy <= 0.0) {
                  e.complete = true;
                  break;
               }

               if (hit.hit) {
                  int oldRegion = e.region;
                  bool transmitted = applyBarrier(e, cfg.mats, hit.nextRegion, hit.normalTowardNext);
                  if (transmitted && oldRegion != REGION_VAC && e.region == REGION_VAC) {
                     recordEscape(e, cfg, pixel);
                     e.complete = true;
                  }
                  continue;
               }

               if (totalRate > 0.0) {
                  double pick = rng.openUniform() * totalRate;
                  if (pick < elasticRate) {
                     scatterElastic(e, mat, cfg.elems, cumulativeElastic, elasticRate, rng);
                  }
                  else {
                     ElectronState secondary;
                     if (scatterInelastic(e, mat, secondary, rng) && stackTop < MAX_SECONDARY_STACK) {
                        secondary.type = (cos(e.theta) > 0.0) ? TYPE_SE1 : TYPE_SE2;
                        stack[stackTop++] = e;
                        e = secondary;
                        atomicAdd(&cfg.genSECounts[pixel], 1);
                     }
                  }
               }
            }

            if (stackTop == 0)
               break;
            e = stack[--stackTop];
         }
      }

      __global__ void compositeImageKernel(KernelConfig cfg)
      {
         int local = blockIdx.x * blockDim.x + threadIdx.x;
         if (local >= cfg.batchTrajectoryCount)
            return;

         int globalTrajectory = cfg.startTrajectory + local;
         int pixel = globalTrajectory / cfg.trajPerPixel;
         if (pixel >= cfg.pixelCount)
            return;

         runOneTrajectory(cfg, pixel, (unsigned long long)globalTrajectory);
      }

      bool checkCuda(cudaError_t err, const char* op)
      {
         if (err == cudaSuccess)
            return true;
         std::printf("CompositeImageGPU: CUDA error during %s: %s\n", op, cudaGetErrorString(err));
         std::fflush(stdout);
         return false;
      }
   }

   bool isAvailable()
   {
      int deviceCount = 0;
      return cudaGetDeviceCount(&deviceCount) == cudaSuccess && deviceCount > 0;
   }

   bool run(const GPURunConfig& cfg, GPUOutput& out)
   {
      if (cfg.nx <= 0 || cfg.ny <= 0 || cfg.trajPerPixel <= 0 || cfg.elems.empty())
         return false;

      const int pixelCount = cfg.nx * cfg.ny;
      const int totalTrajectories = pixelCount * cfg.trajPerPixel;
      const long long histLen = (cfg.histEnabled && cfg.histNEbins > 0 && cfg.histNBbins > 0)
         ? (long long)pixelCount * cfg.histNEbins * cfg.histNBbins : 0;
      out.seYield.assign(pixelCount, 0.0);
      out.se1Yield.assign(pixelCount, 0.0);
      out.se2Yield.assign(pixelCount, 0.0);
      out.bseYield.assign(pixelCount, 0.0);
      out.genSeYield.assign(pixelCount, 0.0);

      MatGPU* dMats = nullptr;
      ElemTableGPU* dElems = nullptr;
      double* dPixelX = nullptr;
      double* dPixelY = nullptr;
      int* dSE = nullptr;
      int* dSE1 = nullptr;
      int* dSE2 = nullptr;
      int* dTotal = nullptr;
      int* dGenSE = nullptr;
      int* dHist = nullptr;

      bool ok = true;
      ok = ok && checkCuda(cudaMalloc((void**)&dMats, sizeof(MatGPU) * 4), "cudaMalloc mats");
      ok = ok && checkCuda(cudaMalloc((void**)&dElems, sizeof(ElemTableGPU) * cfg.elems.size()), "cudaMalloc elems");
      ok = ok && checkCuda(cudaMalloc((void**)&dPixelX, sizeof(double) * pixelCount), "cudaMalloc pixelX");
      ok = ok && checkCuda(cudaMalloc((void**)&dPixelY, sizeof(double) * pixelCount), "cudaMalloc pixelY");
      ok = ok && checkCuda(cudaMalloc((void**)&dSE, sizeof(int) * pixelCount), "cudaMalloc seCounts");
      ok = ok && checkCuda(cudaMalloc((void**)&dSE1, sizeof(int) * pixelCount), "cudaMalloc se1Counts");
      ok = ok && checkCuda(cudaMalloc((void**)&dSE2, sizeof(int) * pixelCount), "cudaMalloc se2Counts");
      ok = ok && checkCuda(cudaMalloc((void**)&dTotal, sizeof(int) * pixelCount), "cudaMalloc totalCounts");
      ok = ok && checkCuda(cudaMalloc((void**)&dGenSE, sizeof(int) * pixelCount), "cudaMalloc genSECounts");
      if (ok && histLen > 0)
         ok = ok && checkCuda(cudaMalloc((void**)&dHist, sizeof(int) * histLen), "cudaMalloc escapeHist");

      if (ok) {
         ok = ok && checkCuda(cudaMemcpy(dMats, cfg.mats, sizeof(MatGPU) * 4, cudaMemcpyHostToDevice), "copy mats");
         ok = ok && checkCuda(cudaMemcpy(dElems, cfg.elems.data(), sizeof(ElemTableGPU) * cfg.elems.size(), cudaMemcpyHostToDevice), "copy elems");
         ok = ok && checkCuda(cudaMemcpy(dPixelX, cfg.pixelX.data(), sizeof(double) * pixelCount, cudaMemcpyHostToDevice), "copy pixelX");
         ok = ok && checkCuda(cudaMemcpy(dPixelY, cfg.pixelY.data(), sizeof(double) * pixelCount, cudaMemcpyHostToDevice), "copy pixelY");
         ok = ok && checkCuda(cudaMemset(dSE, 0, sizeof(int) * pixelCount), "clear seCounts");
         ok = ok && checkCuda(cudaMemset(dSE1, 0, sizeof(int) * pixelCount), "clear se1Counts");
         ok = ok && checkCuda(cudaMemset(dSE2, 0, sizeof(int) * pixelCount), "clear se2Counts");
         ok = ok && checkCuda(cudaMemset(dTotal, 0, sizeof(int) * pixelCount), "clear totalCounts");
         ok = ok && checkCuda(cudaMemset(dGenSE, 0, sizeof(int) * pixelCount), "clear genSECounts");
         if (dHist)
            ok = ok && checkCuda(cudaMemset(dHist, 0, sizeof(int) * histLen), "clear escapeHist");
      }

      if (ok) {
         KernelConfig kcfg;
         kcfg.mats = dMats;
         kcfg.elems = dElems;
         kcfg.pixelX = dPixelX;
         kcfg.pixelY = dPixelY;
         kcfg.seCounts = dSE;
         kcfg.se1Counts = dSE1;
         kcfg.se2Counts = dSE2;
         kcfg.totalCounts = dTotal;
         kcfg.genSECounts = dGenSE;
         kcfg.pixelCount = pixelCount;
         kcfg.trajPerPixel = cfg.trajPerPixel;
         kcfg.geom = cfg.geom;
         kcfg.beamE = cfg.beamE;
         kcfg.beamSizeM = cfg.beamSizeM;
         kcfg.beamStartZ = cfg.beamStartZ;
         kcfg.seThresholdJ = cfg.seThresholdJ;
         kcfg.seed = cfg.seed == 0 ? 0x123456789abcdef0ULL : cfg.seed;
         kcfg.escapeHist = dHist;   // nullptr when histogram disabled
         kcfg.histNEbins = cfg.histNEbins;
         kcfg.histNBbins = cfg.histNBbins;
         kcfg.histEbinWidthJ = cfg.histEbinWidthJ;

         const int threads = 128;
         for (int start = 0; ok && start < totalTrajectories; start += LAUNCH_BATCH_TRAJ) {
            int batch = std::min(LAUNCH_BATCH_TRAJ, totalTrajectories - start);
            kcfg.startTrajectory = start;
            kcfg.batchTrajectoryCount = batch;
            int blocks = (batch + threads - 1) / threads;
            compositeImageKernel<<<blocks, threads>>>(kcfg);
            ok = ok && checkCuda(cudaGetLastError(), "kernel launch");
            ok = ok && checkCuda(cudaDeviceSynchronize(), "kernel synchronize");
         }
      }

      std::vector<int> se(pixelCount, 0), se1(pixelCount, 0), se2(pixelCount, 0), total(pixelCount, 0), genSE(pixelCount, 0);
      if (ok) {
         ok = ok && checkCuda(cudaMemcpy(se.data(), dSE, sizeof(int) * pixelCount, cudaMemcpyDeviceToHost), "copy seCounts");
         ok = ok && checkCuda(cudaMemcpy(se1.data(), dSE1, sizeof(int) * pixelCount, cudaMemcpyDeviceToHost), "copy se1Counts");
         ok = ok && checkCuda(cudaMemcpy(se2.data(), dSE2, sizeof(int) * pixelCount, cudaMemcpyDeviceToHost), "copy se2Counts");
         ok = ok && checkCuda(cudaMemcpy(total.data(), dTotal, sizeof(int) * pixelCount, cudaMemcpyDeviceToHost), "copy totalCounts");
         ok = ok && checkCuda(cudaMemcpy(genSE.data(), dGenSE, sizeof(int) * pixelCount, cudaMemcpyDeviceToHost), "copy genSECounts");
         if (dHist) {
            out.escapeHist.assign((size_t)histLen, 0);
            ok = ok && checkCuda(cudaMemcpy(out.escapeHist.data(), dHist, sizeof(int) * histLen, cudaMemcpyDeviceToHost), "copy escapeHist");
            out.histNEbins = cfg.histNEbins;
            out.histNBbins = cfg.histNBbins;
         }
      }

      cudaFree(dMats);
      cudaFree(dElems);
      cudaFree(dPixelX);
      cudaFree(dPixelY);
      cudaFree(dSE);
      cudaFree(dSE1);
      cudaFree(dSE2);
      cudaFree(dTotal);
      cudaFree(dGenSE);
      cudaFree(dHist);

      if (!ok)
         return false;

      const double denom = (double)cfg.trajPerPixel;
      for (int i = 0; i < pixelCount; ++i) {
         out.seYield[i]    = se[i]    / denom;
         out.se1Yield[i]   = se1[i]   / denom;
         out.se2Yield[i]   = se2[i]   / denom;
         out.bseYield[i]   = (total[i] - se[i]) / denom;
         out.genSeYield[i] = genSE[i] / denom;
      }
      return true;
   }
}
