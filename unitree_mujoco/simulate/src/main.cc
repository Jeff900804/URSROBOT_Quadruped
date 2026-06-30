// Copyright 2021 DeepMind Technologies Limited
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// !!! hack code: make glfw_adapter.window_ public
#define private public
#include "glfw_adapter.h"
#undef private

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <mutex>
#include <new>
#include <string>
#include <thread>

#include <mujoco/mujoco.h>
#include "simulate.h"
#include "array_safety.h"
#include "unitree_sdk2_bridge.h"
#include "param.h"

#define MUJOCO_PLUGIN_DIR "mujoco_plugin"
#define NUM_MOTOR_IDL_GO 20

extern "C"
{
#if defined(_WIN32) || defined(__CYGWIN__)
#include <windows.h>
#else
#if defined(__APPLE__)
#include <mach-o/dyld.h>
#endif
#include <sys/errno.h>
#include <unistd.h>
#endif
}

#include <cstdio>
#include "mujoco/mujoco.h"

// simulate.cpp (unitree_mujoco)
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <vector>
#include <algorithm>  // std::clamp

#include <cmath>
#include <string>
#include <initializer_list>

// ------------------------------
// Robot state logger (CSV)
// ------------------------------
#include <limits>

FILE* g_robot_csv = nullptr;

int g_base_body = -1;

// feet: we will USE GEOMS (because your model has only site "imu")
int g_foot_geom[4] = {-1, -1, -1, -1};   // FL, FR, RL, RR
std::recursive_mutex* g_mtx_ptr = nullptr;

// Utility: try multiple names and return first found id (or -1)
static int find_first_id(const mjModel* m, mjtObj obj,
                         std::initializer_list<const char*> names) {
  for (auto n : names) {
    int id = mj_name2id(m, obj, n);
    if (id >= 0) return id;
  }
  return -1;
}

// Convert rotation matrix (row-major 3x3 from d->xmat) to roll/pitch/yaw (ZYX)
static inline void mat_to_rpy(const mjtNum* R9, double& roll, double& pitch, double& yaw) {
  const double r00 = (double)R9[0], r01 = (double)R9[1], r02 = (double)R9[2];
  const double r10 = (double)R9[3], r11 = (double)R9[4], r12 = (double)R9[5];
  const double r20 = (double)R9[6], r21 = (double)R9[7], r22 = (double)R9[8];

  yaw   = std::atan2(r10, r00);
  pitch = std::atan2(-r20, std::sqrt(r21*r21 + r22*r22));
  roll  = std::atan2(r21, r22);
}

// Optional dump (keep your existing if you want)
static void DumpSitesGeoms(const mjModel* m) {
  std::printf("==== SITES (nsite=%d) ====\n", m->nsite);
  for (int i = 0; i < m->nsite; ++i) {
    const char* name = mj_id2name(m, mjOBJ_SITE, i);
    int bid = m->site_bodyid[i];
    const char* bname = mj_id2name(m, mjOBJ_BODY, bid);
    std::printf("site[%4d] name=%s  body=%s(%d)\n", i, name ? name : "(null)", bname ? bname : "(null)", bid);
  }

  std::printf("==== GEOMS (ngeom=%d) ====\n", m->ngeom);
  for (int i = 0; i < m->ngeom; ++i) {
    const char* name = mj_id2name(m, mjOBJ_GEOM, i);
    int bid = m->geom_bodyid[i];
    const char* bname = mj_id2name(m, mjOBJ_BODY, bid);
    std::printf("geom[%4d] name=%s  body=%s(%d)\n", i, name ? name : "(null)", bname ? bname : "(null)", bid);
  }
}

static void InitRobotStateLogger(const mjModel* m) {
  // base body (try common names)
  g_base_body = find_first_id(m, mjOBJ_BODY, {
    "torso_link", "base_link", "trunk", "pelvis", "body"
  });

  // IMPORTANT: your dump shows foot geoms names are exactly: FL FR RL RR
  // so we use these first.
  g_foot_geom[0] = find_first_id(m, mjOBJ_GEOM, {"FL", "fl"});
  g_foot_geom[1] = find_first_id(m, mjOBJ_GEOM, {"FR", "fr"});
  g_foot_geom[2] = find_first_id(m, mjOBJ_GEOM, {"RL", "rl"});
  g_foot_geom[3] = find_first_id(m, mjOBJ_GEOM, {"RR", "rr"});

  std::printf("[robot_log] base_body=%d (%s)\n", g_base_body,
              g_base_body >= 0 ? (mj_id2name(m, mjOBJ_BODY, g_base_body) ?: "(noname)") : "(not found)");
  std::printf("[robot_log] foot_geom FL/FR/RL/RR = %d %d %d %d\n",
              g_foot_geom[0], g_foot_geom[1], g_foot_geom[2], g_foot_geom[3]);

  // sanity check
  for (int k = 0; k < 4; ++k) {
    if (g_foot_geom[k] < 0) {
      std::printf("[robot_log] ERROR: foot geom %d not found. Check XML geom names.\n", k);
    }
  }

  g_robot_csv = std::fopen("robot_state.csv", "w");
  if (!g_robot_csv) {
    std::perror("[robot_log] fopen robot_state.csv failed");
    return;
  }

  std::fprintf(g_robot_csv,
    "time,"
    "base_x,base_y,base_z,roll,pitch,yaw,"
    "fl_x,fl_y,fl_z,fr_x,fr_y,fr_z,rl_x,rl_y,rl_z,rr_x,rr_y,rr_z,"
    "fl_Fx,fl_Fy,fl_Fz,fr_Fx,fr_Fy,fr_Fz,rl_Fx,rl_Fy,rl_Fz,rr_Fx,rr_Fy,rr_Fz,"
    "fl_contact,fr_contact,rl_contact,rr_contact\n"
  );
}

static void CloseRobotStateLogger() {
  if (g_robot_csv) { std::fclose(g_robot_csv); g_robot_csv = nullptr; }
}

// helper: geom world pos
static inline void get_geom_pos(const mjData* d, int gid, double& x, double& y, double& z) {
  if (gid < 0) { x=y=z=std::numeric_limits<double>::quiet_NaN(); return; }
  const mjtNum* p = d->geom_xpos + 3*gid;
  x = (double)p[0]; y = (double)p[1]; z = (double)p[2];
}

// Accumulate per-foot contact forces in WORLD frame by rotating contact-frame forces using con.frame
static void AccumulateFootContactForcesWorld(const mjModel* m, const mjData* d,
                                             double Fw[4][3], int C[4]) {
  for (int i = 0; i < 4; ++i) {
    Fw[i][0] = Fw[i][1] = Fw[i][2] = 0.0;
    C[i] = 0;
  }

  for (int ci = 0; ci < d->ncon; ++ci) {
    const mjContact& con = d->contact[ci];

    // figure which foot
    int foot = -1;
    for (int k = 0; k < 4; ++k) {
      const int gid = g_foot_geom[k];
      if (gid >= 0 && (con.geom1 == gid || con.geom2 == gid)) { foot = k; break; }
    }
    if (foot < 0) continue;

    mjtNum f6[6];
    mj_contactForce(m, d, ci, f6);  // f6[0:3] = force in contact frame (n,t1,t2)

    // con.frame: 3 world vectors (normal, tangent1, tangent2), each 3 dims
    const mjtNum* fr = con.frame;

    // world force = f0*n + f1*t1 + f2*t2
    const double Fx = (double)(f6[0]*fr[0] + f6[1]*fr[3] + f6[2]*fr[6]);
    const double Fy = (double)(f6[0]*fr[1] + f6[1]*fr[4] + f6[2]*fr[7]);
    const double Fz = (double)(f6[0]*fr[2] + f6[1]*fr[5] + f6[2]*fr[8]);

    Fw[foot][0] += Fx;
    Fw[foot][1] += Fy;
    Fw[foot][2] += Fz;
    C[foot] += 1;
  }
}

static void LogRobotState(const mjModel* m, const mjData* d) {
  if (!g_robot_csv) return;
  if (g_base_body < 0) return;

  // base pose
  const mjtNum* bp = d->xpos + 3*g_base_body;
  const mjtNum* R  = d->xmat + 9*g_base_body;
  double roll, pitch, yaw;
  mat_to_rpy(R, roll, pitch, yaw);

  // feet positions (GEOM-based)
  double fp[4][3];
  for (int k = 0; k < 4; ++k) {
    get_geom_pos(d, g_foot_geom[k], fp[k][0], fp[k][1], fp[k][2]);
  }

  // contact forces (WORLD)
  double Fw[4][3];
  int C[4];
  AccumulateFootContactForcesWorld(m, d, Fw, C);

  std::fprintf(g_robot_csv,
    "%.6f,"
    "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
    "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
    "%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,%.6f,"
    "%d,%d,%d,%d\n",
    (double)d->time,
    (double)bp[0], (double)bp[1], (double)bp[2], roll, pitch, yaw,
    fp[0][0], fp[0][1], fp[0][2],
    fp[1][0], fp[1][1], fp[1][2],
    fp[2][0], fp[2][1], fp[2][2],
    fp[3][0], fp[3][1], fp[3][2],
    Fw[0][0], Fw[0][1], Fw[0][2],
    Fw[1][0], Fw[1][1], Fw[1][2],
    Fw[2][0], Fw[2][1], Fw[2][2],
    Fw[3][0], Fw[3][1], Fw[3][2],
    (C[0] > 0), (C[1] > 0), (C[2] > 0), (C[3] > 0)
  );

  // 若你想即時看到檔案更新（debug），打開這行；正式跑長時間請關掉
  // std::fflush(g_robot_csv);
}




static int g_h_fd = -1;

static void InitHeightBinWriter(int dim) {
  if (g_h_fd >= 0) { ::close(g_h_fd); g_h_fd = -1; }   // 防止重入漏 fd

  g_h_fd = ::open("/home/jeff/mujoco_shared/height_scanner_mujoco_f32.bin", O_CREAT | O_TRUNC | O_WRONLY, 0666);

  if (g_h_fd < 0) { perror("open /tmp/height_scanner_mujoco_f32.bin"); return; }

  int32_t d = dim;
  const ssize_t w = ::write(g_h_fd, &d, sizeof(d));
  if (w != (ssize_t)sizeof(d)) {
    perror("write header dim");
  }
  ::fsync(g_h_fd);
}


static void WriteHeightBin(const mjtNum* buf, int dim) {

  if (g_h_fd < 0) return;

  std::vector<float> tmp(dim);
  for (int i = 0; i < dim; ++i) tmp[i] = (float)buf[i];

  const off_t off = sizeof(int32_t);
  const ssize_t want = (ssize_t)(dim * sizeof(float));
  const ssize_t got  = ::pwrite(g_h_fd, tmp.data(), want, off);
  if (got != want) {
    perror("pwrite height data");
  }

  // 測試期保險：確保 controller 即時讀到；跑順後可拿掉
  ::fsync(g_h_fd);
}



// 這個 callback 只拿來印出載了哪些 .so
void PluginReport(const char* filename, int plugin_count, int load_result) {
  std::printf("[plugin] loaded: %s  (count=%d, result=%d)\n",
              filename, plugin_count, load_result);
}

class ElasticBand
{
public:
  ElasticBand(){};
  void Advance(std::vector<double> x, std::vector<double> dx)
  {
    std::vector<double> delta_x = {0.0, 0.0, 0.0};
    delta_x[0] = point_[0] - x[0];
    delta_x[1] = point_[1] - x[1];
    delta_x[2] = point_[2] - x[2];
    double distance = sqrt(delta_x[0] * delta_x[0] + delta_x[1] * delta_x[1] + delta_x[2] * delta_x[2]);

    std::vector<double> direction = {0.0, 0.0, 0.0};
    direction[0] = delta_x[0] / distance;
    direction[1] = delta_x[1] / distance;
    direction[2] = delta_x[2] / distance;

    double v = dx[0] * direction[0] + dx[1] * direction[1] + dx[2] * direction[2];

    f_[0] = (stiffness_ * (distance - length_) - damping_ * v) * direction[0];
    f_[1] = (stiffness_ * (distance - length_) - damping_ * v) * direction[1];
    f_[2] = (stiffness_ * (distance - length_) - damping_ * v) * direction[2];
  }


  double stiffness_ = 200;
  double damping_ = 100;
  std::vector<double> point_ = {0, 0, 3};
  double length_ = 0.0;
  bool enable_ = true;
  std::vector<double> f_ = {0, 0, 0};
};
inline ElasticBand elastic_band;


namespace
{
  namespace mj = ::mujoco;
  namespace mju = ::mujoco::sample_util;
  mj::Simulate* g_sim_ptr = nullptr;

  // constants
  const double syncMisalign = 0.1;       // maximum mis-alignment before re-sync (simulation seconds)
  const double simRefreshFraction = 0.7; // fraction of refresh available for simulation
  const int kErrorLength = 1024;         // load error string length

  // model and data
  mjModel *m = nullptr;
  mjData *d = nullptr;

  // control noise variables
  mjtNum *ctrlnoise = nullptr;
  
  int g_height_sid  = -1;
  int g_height_adr  = 0; 
  int g_height_dim  = 0;
  FILE* g_height_csv = nullptr;

  // 初始化：找到 sensor 的位置並開啟 CSV 檔
void InitHeightScanner(const mjModel* m, const mjData* d) {
  g_height_sid = mj_name2id(m, mjOBJ_SENSOR, "height_scanner");
  if (g_height_sid < 0) {
    std::printf("[height_scanner] sensor not found!\n");
    return;
  }

  g_height_adr = m->sensor_adr[g_height_sid];
  g_height_dim = m->sensor_dim[g_height_sid];

  std::printf("[height_scanner] sid=%d adr=%d dim=%d\n",
              g_height_sid, g_height_adr, g_height_dim);

  // ✅ 先開 bin writer（最重要）
  InitHeightBinWriter(g_height_dim);

  // CSV 可選，失敗也不要 return
  g_height_csv = std::fopen("height_scanner.csv", "w");
  if (!g_height_csv) {
    std::perror("[height_scanner] fopen csv failed (ok, bin still enabled)");
    return;
  }

  std::fprintf(g_height_csv, "time");
  for (int i = 0; i < g_height_dim; ++i) std::fprintf(g_height_csv, ",h%d", i);
  std::fprintf(g_height_csv, "\n");
}

void LogHeightScanner(const mjModel* m, const mjData* d) {
  if (g_height_sid < 0) return;

  const mjtNum* buf = d->sensordata + g_height_adr;

  // ===== CSV：寫「跟 obs 一樣」的處理後數值 =====
  if (g_height_csv) {
    // 1) 先把原始 buf 拷貝成 float（對應你 controller 的 raw）
    std::vector<float> raw(g_height_dim);
    for (int i = 0; i < g_height_dim; ++i) raw[i] = (float)buf[i];

    // 2) 做你 controller 的方向翻轉（你貼的：Nx=17, Ny=11，上下 row swap）
    static constexpr int Nx = 17;  // 每列 17
    static constexpr int Ny = 11;  // 共 11 列
    if (g_height_dim == Nx * Ny) {
      for (int y = 0; y < Ny / 2; ++y) {
        int top = y * Nx;
        int bot = (Ny - 1 - y) * Nx;
        for (int x = 0; x < Nx; ++x) {
          std::swap(raw[top + x], raw[bot + x]);
        }
      }
    }

    // 3) 套用跟 obs 一樣的 (-raw + offset) + clamp
    float offset = 0.3f;  // ⚠️ 這個要跟你 controller params["offset"] 一致
    for (int i = 0; i < g_height_dim; ++i) {
      float v = -raw[i] + offset;
      raw[i] = std::clamp(v, -10.0f, 10.0f);
    }

    // 4) 寫 CSV（寫處理後 raw）
    std::fprintf(g_height_csv, "%.6f", d->time);
    for (int i = 0; i < g_height_dim; ++i) {
      std::fprintf(g_height_csv, ",%.6f", (double)raw[i]);
    }
    std::fprintf(g_height_csv, "\n");
  }

  // ===== BIN：完全不動，照舊寫原始 sensordata =====
  WriteHeightBin(buf, g_height_dim);
}

// 結束時關掉檔案
void CloseHeightScanner() {
  if (g_height_csv) { std::fclose(g_height_csv); g_height_csv = nullptr; }
  if (g_h_fd >= 0) { ::close(g_h_fd); g_h_fd = -1; }
}



  using Seconds = std::chrono::duration<double>;

  //---------------------------------------- plugin handling -----------------------------------------

  // return the path to the directory containing the current executable
  // used to determine the location of auto-loaded plugin libraries
  std::string getExecutableDir()
  {
#if defined(_WIN32) || defined(__CYGWIN__)
    constexpr char kPathSep = '\\';
    std::string realpath = [&]() -> std::string
    {
      std::unique_ptr<char[]> realpath(nullptr);
      DWORD buf_size = 128;
      bool success = false;
      while (!success)
      {
        realpath.reset(new (std::nothrow) char[buf_size]);
        if (!realpath)
        {
          std::cerr << "cannot allocate memory to store executable path\n";
          return "";
        }

        DWORD written = GetModuleFileNameA(nullptr, realpath.get(), buf_size);
        if (written < buf_size)
        {
          success = true;
        }
        else if (written == buf_size)
        {
          // realpath is too small, grow and retry
          buf_size *= 2;
        }
        else
        {
          std::cerr << "failed to retrieve executable path: " << GetLastError() << "\n";
          return "";
        }
      }
      return realpath.get();
    }();
#else
    constexpr char kPathSep = '/';
#if defined(__APPLE__)
    std::unique_ptr<char[]> buf(nullptr);
    {
      std::uint32_t buf_size = 0;
      _NSGetExecutablePath(nullptr, &buf_size);
      buf.reset(new char[buf_size]);
      if (!buf)
      {
        std::cerr << "cannot allocate memory to store executable path\n";
        return "";
      }
      if (_NSGetExecutablePath(buf.get(), &buf_size))
      {
        std::cerr << "unexpected error from _NSGetExecutablePath\n";
      }
    }
    const char *path = buf.get();
#else
    const char *path = "/proc/self/exe";
#endif
    std::string realpath = [&]() -> std::string
    {
      std::unique_ptr<char[]> realpath(nullptr);
      std::uint32_t buf_size = 128;
      bool success = false;
      while (!success)
      {
        realpath.reset(new (std::nothrow) char[buf_size]);
        if (!realpath)
        {
          std::cerr << "cannot allocate memory to store executable path\n";
          return "";
        }

        std::size_t written = readlink(path, realpath.get(), buf_size);
        if (written < buf_size)
        {
          realpath.get()[written] = '\0';
          success = true;
        }
        else if (written == -1)
        {
          if (errno == EINVAL)
          {
            // path is already not a symlink, just use it
            return path;
          }

          std::cerr << "error while resolving executable path: " << strerror(errno) << '\n';
          return "";
        }
        else
        {
          // realpath is too small, grow and retry
          buf_size *= 2;
        }
      }
      return realpath.get();
    }();
#endif

    if (realpath.empty())
    {
      return "";
    }

    for (std::size_t i = realpath.size() - 1; i > 0; --i)
    {
      if (realpath.c_str()[i] == kPathSep)
      {
        return realpath.substr(0, i);
      }
    }

    // don't scan through the entire file system's root
    return "";
  }

  // scan for libraries in the plugin directory to load additional plugins
  void scanPluginLibraries()
  {
    // check and print plugins that are linked directly into the executable
    int nplugin = mjp_pluginCount();
    if (nplugin)
    {
      std::printf("Built-in plugins:\n");
      for (int i = 0; i < nplugin; ++i)
      {
        std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
      }
    }
   

    // define platform-specific strings
#if defined(_WIN32) || defined(__CYGWIN__)
    const std::string sep = "\\";
#else
    const std::string sep = "/";
#endif

    // try to open the ${EXECDIR}/plugin directory
    // ${EXECDIR} is the directory containing the simulate binary itself
    const std::string executable_dir = getExecutableDir();
    if (executable_dir.empty())
    {
      return;
    }

    const std::string plugin_dir = getExecutableDir() + sep + MUJOCO_PLUGIN_DIR;
    mj_loadAllPluginLibraries(
        plugin_dir.c_str(), +[](const char *filename, int first, int count)
                            {
        std::printf("Plugins registered by library '%s':\n", filename);
        for (int i = first; i < first + count; ++i) {
          std::printf("    %s\n", mjp_getPluginAtSlot(i)->name);
        } });
  }

  //------------------------------------------- simulation -------------------------------------------

  mjModel *LoadModel(const char *file, mj::Simulate &sim)
  {
    // this copy is needed so that the mju::strlen call below compiles
    char filename[mj::Simulate::kMaxFilenameLength];
    mju::strcpy_arr(filename, file);

    // make sure filename is not empty
    if (!filename[0])
    {
      return nullptr;
    }

    // load and compile
    char loadError[kErrorLength] = "";
    mjModel *mnew = 0;
    if (mju::strlen_arr(filename) > 4 &&
        !std::strncmp(filename + mju::strlen_arr(filename) - 4, ".mjb",
                      mju::sizeof_arr(filename) - mju::strlen_arr(filename) + 4))
    {
      mnew = mj_loadModel(filename, nullptr);
      if (!mnew)
      {
        mju::strcpy_arr(loadError, "could not load binary model");
      }
    }
    else
    {
      mnew = mj_loadXML(filename, nullptr, loadError, kErrorLength);
      // remove trailing newline character from loadError
      if (loadError[0])
      {
        int error_length = mju::strlen_arr(loadError);
        if (loadError[error_length - 1] == '\n')
        {
          loadError[error_length - 1] = '\0';
        }
      }
    }

    mju::strcpy_arr(sim.load_error, loadError);

    if (!mnew)
    {
      std::printf("%s\n", loadError);
      return nullptr;
    }

    // compiler warning: print and pause
    if (loadError[0])
    {
      // mj_forward() below will print the warning message
      std::printf("Model compiled, but simulation warning (paused):\n  %s\n", loadError);
      sim.run = 0;
    }

    return mnew;
  }

  // simulate in background thread (while rendering in main thread)
  void PhysicsLoop(mj::Simulate &sim)
  {
    // cpu-sim syncronization point
    std::chrono::time_point<mj::Simulate::Clock> syncCPU;
    mjtNum syncSim = 0;

    // ChannelFactory::Instance()->Init(0);
    // UnitreeDds ud(d);

    // run until asked to exit
    while (!sim.exitrequest.load())
    {
      if (sim.droploadrequest.load())
      {
        sim.LoadMessage(sim.dropfilename);
        mjModel *mnew = LoadModel(sim.dropfilename, sim);
        sim.droploadrequest.store(false);

        mjData *dnew = nullptr;
        if (mnew)
          dnew = mj_makeData(mnew);
        if (dnew)
        {
          sim.Load(mnew, dnew, sim.dropfilename);

          mj_deleteData(d);
          mj_deleteModel(m);

          m = mnew;
          d = dnew;
          mj_forward(m, d);

          // allocate ctrlnoise
          free(ctrlnoise);
          ctrlnoise = (mjtNum *)malloc(sizeof(mjtNum) * m->nu);
          mju_zero(ctrlnoise, m->nu);
        }
        else
        {
          sim.LoadMessageClear();
        }
      }

      if (sim.uiloadrequest.load())
      {
        sim.uiloadrequest.fetch_sub(1);
        sim.LoadMessage(sim.filename);
        mjModel *mnew = LoadModel(sim.filename, sim);
        mjData *dnew = nullptr;
        if (mnew)
          dnew = mj_makeData(mnew);
        if (dnew)
        {
          sim.Load(mnew, dnew, sim.filename);

          mj_deleteData(d);
          mj_deleteModel(m);

          m = mnew;
          d = dnew;
          mj_forward(m, d);

          // allocate ctrlnoise
          free(ctrlnoise);
          ctrlnoise = static_cast<mjtNum *>(malloc(sizeof(mjtNum) * m->nu));
          mju_zero(ctrlnoise, m->nu);
        }
        else
        {
          sim.LoadMessageClear();
        }
      }

      // sleep for 1 ms or yield, to let main thread run
      //  yield results in busy wait - which has better timing but kills battery life
      if (sim.run && sim.busywait)
      {
        std::this_thread::yield();
      }
      else
      {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }

      {
        // lock the sim mutex
        const std::unique_lock<std::recursive_mutex> lock(sim.mtx);

        // run only if model is present
        if (m)
        {
          // running
          if (sim.run)
          {
            bool stepped = false;

            // record cpu time at start of iteration
            const auto startCPU = mj::Simulate::Clock::now();

            // elapsed CPU and simulation time since last sync
            const auto elapsedCPU = startCPU - syncCPU;
            double elapsedSim = d->time - syncSim;

            // inject noise
            if (sim.ctrl_noise_std)
            {
              // convert rate and scale to discrete time (Ornstein–Uhlenbeck)
              mjtNum rate = mju_exp(-m->opt.timestep / mju_max(sim.ctrl_noise_rate, mjMINVAL));
              mjtNum scale = sim.ctrl_noise_std * mju_sqrt(1 - rate * rate);

              for (int i = 0; i < m->nu; i++)
              {
                // update noise
                ctrlnoise[i] = rate * ctrlnoise[i] + scale * mju_standardNormal(nullptr);

                // apply noise
                d->ctrl[i] = ctrlnoise[i];
              }
            }

            // requested slow-down factor
            double slowdown = 100 / sim.percentRealTime[sim.real_time_index];

            // misalignment condition: distance from target sim time is bigger than syncmisalign
            bool misaligned =
                mju_abs(Seconds(elapsedCPU).count() / slowdown - elapsedSim) > syncMisalign;

            // out-of-sync (for any reason): reset sync times, step
            if (elapsedSim < 0 || elapsedCPU.count() < 0 || syncCPU.time_since_epoch().count() == 0 ||
                misaligned || sim.speed_changed)
            {
              // re-sync
              syncCPU = startCPU;
              syncSim = d->time;
              sim.speed_changed = false;

              // run single step, let next iteration deal with timing
              mj_step(m, d);
              stepped = true;
            }

            // in-sync: step until ahead of cpu
            else
            {
              bool measured = false;
              mjtNum prevSim = d->time;

              double refreshTime = simRefreshFraction / sim.refresh_rate;

              // step while sim lags behind cpu and within refreshTime
              while (Seconds((d->time - syncSim) * slowdown) < mj::Simulate::Clock::now() - syncCPU &&
                     mj::Simulate::Clock::now() - startCPU < Seconds(refreshTime))
              {
                // measure slowdown before first step
                if (!measured && elapsedSim)
                {
                  sim.measured_slowdown =
                      std::chrono::duration<double>(elapsedCPU).count() / elapsedSim;
                  measured = true;
                }

                // elastic band on base link
                if (param::config.enable_elastic_band == 1)
                {
                  if (elastic_band.enable_)
                  {
                    std::vector<double> x = {d->qpos[0], d->qpos[1], d->qpos[2]};
                    std::vector<double> dx = {d->qvel[0], d->qvel[1], d->qvel[2]};

                    elastic_band.Advance(x, dx);

                    d->xfrc_applied[param::config.band_attached_link] = elastic_band.f_[0];
                    d->xfrc_applied[param::config.band_attached_link + 1] = elastic_band.f_[1];
                    d->xfrc_applied[param::config.band_attached_link + 2] = elastic_band.f_[2];
                  }
                }

                // call mj_step
                mj_step(m, d);
                stepped = true;

                // break if reset
                if (d->time < prevSim)
                {
                  break;
                }
              }
            }

            // save current state to history buffer
            if (stepped)
            {
              sim.AddToHistory();
              // 把這一步的 height_scanner 數值寫進 CSV
              LogHeightScanner(m, d);
              LogRobotState(m, d);
            }
          }

          // paused
          else
          {
            // run mj_forward, to update rendering and joint sliders
            mj_forward(m, d);
            sim.speed_changed = true;
          }
        }
      } // release std::lock_guard<std::mutex>
    }
  }
} // namespace

//-------------------------------------- physics_thread --------------------------------------------

void PhysicsThread(mj::Simulate *sim, const char *filename)
{
  // request loadmodel if file given (otherwise drag-and-drop)
  if (filename != nullptr)
  {
    sim->LoadMessage(filename);
    m = LoadModel(filename, *sim);
    if (m)
      d = mj_makeData(m);
    if (d)
    {
      sim->Load(m, d, filename);
      mj_resetDataKeyframe(m, d, 0);   // home
      mj_forward(m, d);

      mj_forward(m, d);
      
      std::printf("[debug] nkey=%d\n", m->nkey);
      for (int i = 0; i < m->nkey; ++i) {
        const char* kname = mj_id2name(m, mjOBJ_KEY, i);
        std::printf("[debug] key[%d]=%s\n", i, kname ? kname : "(null)");
      }
      
      // 初始化 height_scanner CSV
      InitHeightScanner(m, d);
      InitRobotStateLogger(m);
      DumpSitesGeoms(m);

      // allocate ctrlnoise
      free(ctrlnoise);
      ctrlnoise = static_cast<mjtNum *>(malloc(sizeof(mjtNum) * m->nu));
      mju_zero(ctrlnoise, m->nu);
    }
    else
    {
      sim->LoadMessageClear();
    }
  }

  PhysicsLoop(*sim);

  // delete everything we allocated
  CloseHeightScanner();   
  CloseRobotStateLogger();
  free(ctrlnoise);
  mj_deleteData(d);
  mj_deleteModel(m);

  exit(0);
}

void *UnitreeSdk2BridgeThread(void *arg)
{
  // Wait for mujoco data
  while (true)
  {
    if (d)
    {
      std::cout << "Mujoco data is prepared" << std::endl;
      break;
    }
    usleep(500000);
  }

  unitree::robot::ChannelFactory::Instance()->Init(param::config.domain_id, param::config.interface);


  int body_id = mj_name2id(m, mjOBJ_BODY, "torso_link");
  if (body_id < 0) {
    body_id = mj_name2id(m, mjOBJ_BODY, "base_link");
  }
  param::config.band_attached_link = 6 * body_id;
  
  std::unique_ptr<UnitreeSDK2BridgeBase> interface = nullptr;
  if (m->nu > NUM_MOTOR_IDL_GO) {
    interface = std::make_unique<G1Bridge>(m, d);
  } else {
    interface = std::make_unique<Go2Bridge>(m, d);
  }
  interface->start();
  
  while (true)
  {
    sleep(1);
  }
}
//------------------------------------------ main --------------------------------------------------

// machinery for replacing command line error by a macOS dialog box when running under Rosetta
#if defined(__APPLE__) && defined(__AVX__)
extern void DisplayErrorDialogBox(const char *title, const char *msg);
static const char *rosetta_error_msg = nullptr;
__attribute__((used, visibility("default"))) extern "C" void _mj_rosettaError(const char *msg)
{
  rosetta_error_msg = msg;
}
#endif

// user keyboard callback
void user_key_cb(GLFWwindow* window, int key, int scancode, int act, int mods) {
  if (act != GLFW_PRESS) return;

  if (key == GLFW_KEY_BACKSPACE) {
    if (!g_sim_ptr || !m || !d) return;

    const std::unique_lock<std::recursive_mutex> lock(g_sim_ptr->mtx);

    // 建議：先暫停一下避免 reset 當下又被 step
    g_sim_ptr->run = 0;

    // reset to keyframe 0 ("home")
    mj_resetDataKeyframe(m, d, 0);

    // 清掉外力（彈力帶 / 其它 xfrc_applied）避免 reset 後被殘留外力扯爆
    mju_zero(d->xfrc_applied, 6 * m->nbody);

    // 如果你有 ctrlnoise，清掉它
    if (ctrlnoise) mju_zero(ctrlnoise, m->nu);

    mj_forward(m, d);

    // 如果你想 reset 後立刻繼續跑，再打開
    g_sim_ptr->run = 1;
  }
}



// run event loop
int main(int argc, char **argv)
{
  std::printf("[unitree_mujoco] calling mj_loadAllPluginLibraries...\n");

  mj_loadAllPluginLibraries(
      "/home/jeff/mujoco/unitree_mujoco/simulate/mujoco/bin/mujoco_plugin",PluginReport);

  std::printf("[unitree_mujoco] plugin scan finished.\n");
  // display an error if running on macOS under Rosetta 2
#if defined(__APPLE__) && defined(__AVX__)
  if (rosetta_error_msg)
  {
    DisplayErrorDialogBox("Rosetta 2 is not supported", rosetta_error_msg);
    std::exit(1);
  }
#endif

  // print version, check compatibility
  std::printf("MuJoCo version %s\n", mj_versionString());
  if (mjVERSION_HEADER != mj_version())
  {
    mju_error("Headers and library have different versions");
  }

  // scan for libraries in the plugin directory to load additional plugins
  scanPluginLibraries();

  mjvCamera cam;
  mjv_defaultCamera(&cam);

  mjvOption opt;
  mjv_defaultOption(&opt);

  mjvPerturb pert;
  mjv_defaultPerturb(&pert);

  // Load simulation configuration
  std::filesystem::path proj_dir = std::filesystem::path(getExecutableDir()).parent_path();
  param::config.load_from_yaml(proj_dir / "config.yaml");
  param::helper(argc, argv);
  if(param::config.robot_scene.is_relative()) {
    param::config.robot_scene = proj_dir.parent_path() / "unitree_robots" / param::config.robot / param::config.robot_scene;
  }

  // simulate object encapsulates the UI
  auto sim = std::make_unique<mj::Simulate>(
    std::make_unique<mj::GlfwAdapter>(),
    &cam, &opt, &pert, /* is_passive = */ false);
    
  g_sim_ptr = sim.get();
  g_mtx_ptr = &g_sim_ptr->mtx;

  std::thread unitree_thread(UnitreeSdk2BridgeThread, nullptr);

  // start physics thread
  std::thread physicsthreadhandle(&PhysicsThread, sim.get(), param::config.robot_scene.c_str());
  // start simulation UI loop (blocking call)
  glfwSetKeyCallback(static_cast<mj::GlfwAdapter*>(sim->platform_ui.get())->window_,user_key_cb);
  sim->RenderLoop();
  physicsthreadhandle.join();
 

  pthread_exit(NULL);
  return 0;
}
