#include "writer.h"
#include "graph.h"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <random>
#include <sys/stat.h>

// ─── Helpers ──────────────────────────────────────────────────────────────────

static void ensure_dir(const STR& path) {
    mkdir(path.c_str(), 0755);
}

static STR make_event_path(const STR& dir, const STR& run_name) {
    STR run_dir = dir + "/" + run_name;
    ensure_dir(run_dir);
    // FIX 3: use hostname + PID in filename as TensorBoard 2.x requires
    return tensorboard::logdir(run_dir);
}

// Banner printer
static void print_section(const STR& title) {
    std::cout << "\n";
    std::cout << "## " << title << "...\n";
}

// Generate normally distributed random values
static F64V normal_samples(std::mt19937& rng, int n,
                                           F64 mean, F64 stddev) {
    std::normal_distribution<F64> dist(mean, stddev);
    F64V v(n);
    for (auto& x : v) x = dist(rng);
    return v;
}

void demo_histograms(const STR& logdir) {
    print_section("Demo 3: Histogram Summaries");

    STR path = make_event_path(logdir, "histograms");
    tensorboard::EventWriter writer(path);
    std::cout << "  Writing to: " << path << "\n";

    std::mt19937 rng(123);
    int num_steps = 50;

    for (int step = 0; step < num_steps; ++step) {
        F32 t = static_cast<F32>(step) / num_steps;

        // Layer 1 weights: gradually tightening distribution
        F64 w1_std = 1.0 - 0.7 * t;  // 1.0 → 0.3
        auto w1 = normal_samples(rng, 512, 0.0, w1_std);
        writer.add_histo("weights/layer1", w1, step, 40);

        // Layer 2 weights: bimodal → unimodal
        F64 mix = t; // 0 = bimodal, 1 = unimodal
        F64V w2;
        auto half_a = normal_samples(rng, 256, -1.0 * (1.0 - mix), 0.5);
        auto half_b = normal_samples(rng, 256,  1.0 * (1.0 - mix), 0.5);
        w2.insert(w2.end(), half_a.begin(), half_a.end());
        w2.insert(w2.end(), half_b.begin(), half_b.end());
        writer.add_histo("weights/layer2", w2, step, 40);

        // Activations: shifting mean over training
        F64 act_mean = -2.0 + 4.0 * t; // -2 → +2
        auto acts = normal_samples(rng, 1024, act_mean, 0.8);
        writer.add_histo("activations/relu", acts, step, 40);

        // Gradients: shrinking magnitude (gradient vanishing demo)
        F64 grad_std = 0.5 * std::exp(-3.0 * t);
        auto grads = normal_samples(rng, 256, 0.0, grad_std);
        writer.add_histo("gradients/layer1", grads, step, 30);

        if (step % 10 == 0) {
            std::cout << "  Step " << std::setw(2) << step
                      << " | w1_std=" << std::fixed << std::setprecision(3) << w1_std
                      << " | act_mean=" << std::fixed << std::setprecision(3) << act_mean
                      << " | grad_std=" << std::scientific << std::setprecision(2) << grad_std
                      << "\n";
        }
    }
    std::cout << "  ✓ Wrote " << num_steps
              << " steps of histogram data (weights, activations, gradients)\n";
}
// ─── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    STR logdir = (argc > 1) ? argv[1] : "/tmp/tb_demo";
    ensure_dir(logdir);

    std::cout << ".tfevents files: " << logdir << "\n";
    
    try {
        demo_graph(logdir);
        demo_scalars(logdir);
        demo_images(logdir);
        demo_histograms(logdir);
    } catch (const std::exception& e) {
        std::cerr << "\n  ERROR: " << e.what() << "\n";
        return 1;
    }

    std::cout << "> tensorboard --logdir=" << logdir << "\n";
    return 0;
}
