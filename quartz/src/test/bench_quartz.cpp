/* 
 * Wrapper around Quartz for benchmarking, cribbed from
 * "The T-Complexity Costs of Error Correction for Control Flow in Quantum
 * Computation", https://arxiv.org/abs/2311.12772
 */

#include "quartz/parser/qasm_parser.h"
#include "quartz/tasograph/substitution.h"
#include "quartz/tasograph/tasograph.h"

using namespace quartz;

void parse_args(char **argv, int argc, bool &simulated_annealing,
                bool &early_stop, bool &disable_search,
                std::string &input_filename, std::string &output_filename,
                std::string &eqset_filename, double &timeout) {
  assert(argv[1] != nullptr);
  input_filename = std::string(argv[1]);
  early_stop = true;
  for (int i = 2; i < argc; i++) {
    if (!std::strcmp(argv[i], "--output")) {
      if (i + 1 < argc) {
        output_filename = std::string(argv[++i]);
      }
      continue;
    }
    if (!std::strcmp(argv[i], "--eqset")) {
      if (i + 1 < argc) {
        eqset_filename = std::string(argv[++i]);
      }
      continue;
    }
    if (!std::strcmp(argv[i], "--disable_search")) {
      disable_search = true;
      continue;
    }
    if (!std::strcmp(argv[i], "--timeout")) {
      if (i + 1 < argc) {
        timeout = atof(argv[++i]);
      }
      continue;
    }
  }
}

int main(int argc, char **argv) {
  std::string input_fn, output_fn;
  std::string eqset_fn = "";
  bool simulated_annealing = false;
  bool early_stop = false;
  bool disable_search = false;
  double timeout = 600;
  parse_args(argv, argc, simulated_annealing, early_stop, disable_search,
             input_fn, output_fn, eqset_fn, timeout);
  auto fn = input_fn.substr(input_fn.rfind('/') + 1);

  // Construct contexts
  ParamInfo param_info;
  Context src_ctx({GateType::h, GateType::ccz, GateType::ccx, GateType::x, GateType::cx,
                   GateType::t, GateType::tdg, GateType::s, GateType::sdg,
                   GateType::input_qubit, GateType::input_param},
                  &param_info);
  Context dst_ctx({GateType::h, GateType::x, GateType::t, GateType::tdg, GateType::add,
                   GateType::cx, GateType::input_qubit, GateType::input_param},
                  &param_info);
  auto union_ctx = union_contexts(&src_ctx, &dst_ctx);

  auto xfer_pair = GraphXfer::ccz_cx_t_xfer(&src_ctx, &dst_ctx, &union_ctx);
  // Load qasm file
  QASMParser qasm_parser(&src_ctx);
  CircuitSeq *dag = nullptr;
  if (!qasm_parser.load_qasm(input_fn, dag)) {
    std::cout << "Parser failed" << std::endl;
  }
  Graph graph(&src_ctx, dag);

  auto start = std::chrono::steady_clock::now();
  // Greedy toffoli flip
  auto graph_before_search = graph.toffoli_flip_greedy(
      GateType::rz, xfer_pair.first, xfer_pair.second);
  //   graph_before_search->to_qasm(input_fn + ".toffoli_flip", false, false);

  // Optimization
  std::shared_ptr<Graph> optimized_graph;
  if (disable_search) {
    optimized_graph = graph_before_search;
  } else {
    graph_before_search->optimize(&dst_ctx, eqset_fn, fn, /*print_message=*/
                                  true, nullptr, -1.0, timeout);
  }
  auto end = std::chrono::steady_clock::now();
  std::cout << "Optimization results of Quartz for " << fn
            << " on Clifford+T gate set."
            << " Gate count after optimization: "
            << optimized_graph->gate_count() << ", "
            << "Circuit depth: " << optimized_graph->circuit_depth() << ", "
            << (double)std::chrono::duration_cast<std::chrono::milliseconds>(
                   end - start)
                       .count() /
                   1000.0
            << " seconds." << std::endl;
  optimized_graph->to_qasm(output_fn, false, false);

  return 0;
}

