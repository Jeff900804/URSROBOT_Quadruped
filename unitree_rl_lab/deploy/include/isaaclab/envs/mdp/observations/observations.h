// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "isaaclab/envs/manager_based_rl_env.h"
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cmath>

#include <fstream>
#include <array>

/////////////////////////////////////////////////////四肢末端位置///////////////////////////////////////////////////// 
struct V3 {
  double x, y, z;
};

static inline V3 v3_add(const V3& a, const V3& b) { return {a.x+b.x, a.y+b.y, a.z+b.z}; }

// rotate vector by X axis
static inline V3 rotX(double q, const V3& v) {
  double c = std::cos(q), s = std::sin(q);
  return { v.x, c*v.y - s*v.z, s*v.y + c*v.z };
}

// rotate vector by Y axis
static inline V3 rotY(double q, const V3& v) {
  double c = std::cos(q), s = std::sin(q);
  return { c*v.x + s*v.z, v.y, -s*v.x + c*v.z };
}

struct LegKine {
  V3 p_base_hip;
  V3 p_hip_thigh;
  V3 p_thigh_calf;
  V3 p_calf_foot;
};

static inline V3 fk_go2_leg(double q_abd, double q_hip, double q_knee, const LegKine& K) {
  // p = p_base_hip + Rx(q_abd) * ( p_hip_thigh + Ry(q_hip) * ( p_thigh_calf + Ry(q_knee)*p_calf_foot ) )
  V3 t3 = rotY(q_knee, K.p_calf_foot);
  V3 t2 = v3_add(K.p_thigh_calf, t3);
  V3 t1 = rotY(q_hip, t2);
  V3 t0 = v3_add(K.p_hip_thigh, t1);
  V3 t  = rotX(q_abd, t0);
  return v3_add(K.p_base_hip, t);
}

static inline std::array<LegKine,4> go2_leg_params() {
  // 0=FL,1=FR,2=RL,3=RR
  std::array<LegKine,4> K;

  // 直接用你貼的 go2.xml
  V3 p_thigh_calf{0.0, 0.0, -0.213};
  V3 p_calf_foot {0.0, 0.0, -0.213};

  K[0] = { V3{ 0.1934,  0.0465, 0.0}, V3{0.0,  0.0955, 0.0}, p_thigh_calf, p_calf_foot }; // FL
  K[1] = { V3{ 0.1934, -0.0465, 0.0}, V3{0.0, -0.0955, 0.0}, p_thigh_calf, p_calf_foot }; // FR
  K[2] = { V3{-0.1934,  0.0465, 0.0}, V3{0.0,  0.0955, 0.0}, p_thigh_calf, p_calf_foot }; // RL
  K[3] = { V3{-0.1934, -0.0465, 0.0}, V3{0.0, -0.0955, 0.0}, p_thigh_calf, p_calf_foot }; // RR

  return K;
}

static inline void append_feet_fk_csv(
  isaaclab::ManagerBasedRLEnv* env,
  const std::array<V3,4>& feet_base   // 0=FL,1=FR,2=RL,3=RR
){
  static bool inited = false;
  static std::ofstream ofs;
  static long long step = 0;

  // 可用 params["csv_path"] 覆寫
  std::string csv_path = "/home/jeff/mujoco_shared/feet_fk.csv";
  try { csv_path = env->cfg["logging"]["feet_fk_csv"].as<std::string>(); } catch(...) {}

  if (!inited) {
    ofs.open(csv_path, std::ios::out);
    if (ofs.is_open()) {
      ofs << "t,FL_x,FL_y,FL_z,FR_x,FR_y,FR_z,RL_x,RL_y,RL_z,RR_x,RR_y,RR_z\n";
      ofs.flush();
      inited = true;
    }
  }

  if (!ofs.is_open()) return;

  double t = (double)step * (double)env->step_dt;
  step++;

  const auto& FL = feet_base[0];
  const auto& FR = feet_base[1];
  const auto& RL = feet_base[2];
  const auto& RR = feet_base[3];

  ofs << t << ","
      << -FL.x << "," << -FL.y << "," << -FL.z << ","
      << -FR.x << "," << -FR.y << "," << -FR.z << ","
      << -RL.x << "," << -RL.y << "," << -RL.z << ","
      << -RR.x << "," << -RR.y << "," << -RR.z << "\n";

  // 你可以每 N 行 flush 一次避免太慢
  if ((step % 200) == 0) ofs.flush();
}
///////////////////////////////////////////////////// 


static bool load_height_csv_once_time(
    const char* path,
    int dim_expected,
    std::vector<double>& times,
    std::vector<std::vector<float>>& table)
{
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        std::printf("[hs] CSV open failed: %s\n", path);
        return false;
    }

    times.clear();
    table.clear();
    times.reserve(20000);
    table.reserve(20000);

    std::string line;
    bool first_line = true;

    while (std::getline(ifs, line)) {
        if (line.empty()) continue;

        // 把 tab / comma 都換成空白，統一用 stringstream 讀
        for (char& ch : line) {
            if (ch == '\t' || ch == ',') ch = ' ';
        }

        // skip header：第一行通常是 "time h0 h1 ..."
        if (first_line) {
            first_line = false;
            if (line.find("time") != std::string::npos) continue;
        }

        std::stringstream ss(line);

        double t;
        if (!(ss >> t)) continue;

        std::vector<float> row;
        row.reserve(dim_expected);

        float v;
        while (ss >> v) row.push_back(v);

        if ((int)row.size() != dim_expected) {
            // 有些行可能壞掉或少欄位，跳過
            continue;
        }

        times.push_back(t);
        table.push_back(std::move(row));
    }

    std::printf("[hs] CSV loaded: rows=%zu dim=%d time_range=[%.3f, %.3f]\n",
                table.size(), dim_expected,
                times.empty() ? 0.0 : times.front(),
                times.empty() ? 0.0 : times.back());
    return !table.empty();
}

namespace isaaclab
{
namespace mdp
{

static float g_cmd_gate = 1.0f;
static int   g_gate_hold = 0;   // 觸發後 hold 幾個 step


REGISTER_OBSERVATION(base_ang_vel)
{
    auto & asset = env->robot;
    auto & data = asset->data.root_ang_vel_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(projected_gravity)
{
    auto & asset = env->robot;
    auto & data = asset->data.projected_gravity_b;
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(joint_pos)
{
    auto & asset = env->robot;
    std::vector<float> data;

    std::vector<int> joint_ids;
    try {
        joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
    } catch(const std::exception& e) {
    }

    if(joint_ids.empty())
    {
        data.resize(asset->data.joint_pos.size());
        for(size_t i = 0; i < asset->data.joint_pos.size(); ++i)
        {
            data[i] = asset->data.joint_pos[i];
        }
    }
    else
    {
        data.resize(joint_ids.size());
        for(size_t i = 0; i < joint_ids.size(); ++i)
        {
            data[i] = asset->data.joint_pos[joint_ids[i]];
        }
    }

    return data;
}

REGISTER_OBSERVATION(joint_pos_rel)
{
    auto & asset = env->robot;
    std::vector<float> data;

    data.resize(asset->data.joint_pos.size());
    for(size_t i = 0; i < asset->data.joint_pos.size(); ++i) {
        data[i] = asset->data.joint_pos[i] - asset->data.default_joint_pos[i];
    }

    try {
        std::vector<int> joint_ids;
        joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();
        if(!joint_ids.empty()) {
            std::vector<float> tmp_data;
            tmp_data.resize(joint_ids.size());
            for(size_t i = 0; i < joint_ids.size(); ++i){
                tmp_data[i] = data[joint_ids[i]];
            }
            data = tmp_data;
        }
    } catch(const std::exception& e) {
    
    }

    return data;
}

REGISTER_OBSERVATION(joint_vel_rel)
{
    auto & asset = env->robot;
    auto data = asset->data.joint_vel;

    try {
        const std::vector<int> joint_ids = params["asset_cfg"]["joint_ids"].as<std::vector<int>>();

        if(!joint_ids.empty()) {
            data.resize(joint_ids.size());
            for(size_t i = 0; i < joint_ids.size(); ++i) {
                data[i] = asset->data.joint_vel[joint_ids[i]];
            }
        }
    } catch(const std::exception& e) {
    }
    return std::vector<float>(data.data(), data.data() + data.size());
}

REGISTER_OBSERVATION(last_action)
{
    auto data = env->action_manager->action();
    return std::vector<float>(data.data(), data.data() + data.size());
};

REGISTER_OBSERVATION(velocity_commands)
{
    std::vector<float> obs(3);
    auto & joystick = env->robot->data.joystick;

    const auto cfg = env->cfg["commands"]["base_velocity"]["ranges"];

    obs[0] = std::clamp(joystick->ly(), cfg["lin_vel_x"][0].as<float>(), cfg["lin_vel_x"][1].as<float>());
    obs[1] = std::clamp(-joystick->lx(), cfg["lin_vel_y"][0].as<float>(), cfg["lin_vel_y"][1].as<float>());
    obs[2] = std::clamp(-joystick->rx(), cfg["ang_vel_z"][0].as<float>(), cfg["ang_vel_z"][1].as<float>());
    
    obs[0] *= g_cmd_gate;
    obs[1] *= g_cmd_gate;
    obs[2] *= g_cmd_gate;
    
    return obs;
}

REGISTER_OBSERVATION(gait_phase)
{
    float period = params["period"].as<float>();
    float delta_phase = env->step_dt * (1.0f / period);

    env->global_phase += delta_phase;
    env->global_phase = std::fmod(env->global_phase, 1.0f);

    std::vector<float> obs(2);
    obs[0] = std::sin(env->global_phase * 2 * M_PI);
    obs[1] = std::cos(env->global_phase * 2 * M_PI);
    return obs;
}
// add height_scanner information to observation
#include <cerrno>
#include <cstring>
#include <cstdio>
#include <algorithm>
 
REGISTER_OBSERVATION(height_scanner)
{
//return std::vector<float>(187, 0.12165f);
/*
static bool inited = false;
    static std::vector<double> hs_times;
    static std::vector<std::vector<float>> hs_table;
    static double csv_t0 = 0.0;

    static constexpr int dim = 187;

    if (!inited) {
        const char* csv_path = "/home/jeff/mujoco_shared/height_scanner3.csv";
        inited = load_height_csv_once_time(csv_path, dim, hs_times, hs_table);
        if (!inited) return std::vector<float>(dim, 0.0f);

        csv_t0 = hs_times.front();  // 例如 26.74
    }

    static auto t0 = std::chrono::steady_clock::now();
    double t_sim = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    double t_query = csv_t0 + t_sim;
    
    // 如果你想從 CSV 中間某段開始播，可以加偏移：
    // double extra = 0.0;
    // try { extra = params["start_offset"].as<double>(); } catch(...) {}
    // t_query += extra;

    // 找到第一個 >= t_query 的 index
    auto it = std::lower_bound(hs_times.begin(), hs_times.end(), t_query);
    int idx = 0;
    if (it == hs_times.begin()) idx = 0;
    else if (it == hs_times.end()) idx = (int)hs_times.size() - 1;
    else idx = (int)std::distance(hs_times.begin(), it);

    // 如果你想用「<= t_query 的那列」（sample-hold，避免跳到未來），改成：
    // int idx = (it == hs_times.begin()) ? 0 : (int)std::distance(hs_times.begin(), it) - 1;

    std::vector<float> out = hs_table[idx];

    return out;
    */
    

    // --- 1) 先把 raw 讀出來（你的原本程式） ---
    static int fd = -1;
    static int32_t dim = -1;

    if (fd < 0) {
        fd = ::open("/home/jeff/mujoco_shared/height_scanner_mujoco_f32.bin", O_RDONLY);
        if (fd >= 0) {
            ::pread(fd, &dim, sizeof(dim), 0);
        }
    }
    if (fd < 0 || dim != 187) {
        return std::vector<float>(187, 0.0f);
    }

    std::vector<float> raw(dim);
    const off_t off = sizeof(int32_t);
    const ssize_t n = ::pread(fd, raw.data(), dim * sizeof(float), off);
    if (n != (ssize_t)(dim * sizeof(float))) {
        return std::vector<float>(dim, 0.0f);
    }

    float offset = 0.3f;
    try { offset = params["offset"].as<float>(); } catch(...) {}

    static constexpr int Nx = 17;
    static constexpr int Ny = 11;


    // (你原本的 row flip) mujoco version
    if (dim == Nx * Ny) {
        for (int y = 0; y < Ny / 2; ++y) {
            int top = y * Nx;
            int bot = (Ny - 1 - y) * Nx;
            for (int x = 0; x < Nx; ++x) std::swap(raw[top + x], raw[bot + x]);
        }
    }

    for (int i = 0; i < dim; ++i) {
        float v = -raw[i] + offset;
        raw[i] = std::clamp(v, -10.0f, 10.0f);
    }
/*  
    // PGTT bin 已經是對齊 Isaac/MuJoCo 的 height map：
    for (int i = 0; i < dim; ++i) {
        raw[i] = std::clamp(raw[i], -10.0f, 10.0f);
    }
*/
    // --- 2) FK：把 feet 宣告在外層，避免 scope 問題 ---
    auto& asset = env->robot;
    auto& jp = asset->data.joint_pos;

    std::array<V3,4> feet{};   // 0=FL,1=FR,2=RL,3=RR
    bool have_feet = false;

    if ((int)jp.size() >= 12) {
        double qFL_abd  = jp[0],  qFR_abd  = jp[1],  qRL_abd  = jp[2],  qRR_abd  = jp[3];
        double qFL_hip  = jp[4],  qFR_hip  = jp[5],  qRL_hip  = jp[6],  qRR_hip  = jp[7];
        double qFL_knee = jp[8],  qFR_knee = jp[9],  qRL_knee = jp[10], qRR_knee = jp[11];

        static const auto K = go2_leg_params();
        feet[0] = fk_go2_leg(qFL_abd, qFL_hip, qFL_knee, K[0]);
        feet[1] = fk_go2_leg(qFR_abd, qFR_hip, qFR_knee, K[1]);
        feet[2] = fk_go2_leg(qRL_abd, qRL_hip, qRL_knee, K[2]);
        feet[3] = fk_go2_leg(qRR_abd, qRR_hip, qRR_knee, K[3]);

        have_feet = true;

        // 你要存 log 就在這裡
        append_feet_fk_csv(env, feet);
    }
    return raw;
 
}
}
}
