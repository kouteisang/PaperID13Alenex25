#include <iostream>
#include <cmath>
#include <stdio.h>
#include <string>
#include <fstream>
#include <time.h>
#include <vector>
#include <sstream>

#include <popops/DynamicSlice.hpp>
#include <popops/Sort.hpp>
#include <popops/Reduce.hpp> 
#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poputil/TileMapping.hpp>
#include <poplin/MatMul.hpp>
#include <popops/ElementWise.hpp>
#include <popops/codelets.hpp>
#include <poplin/codelets.hpp>
#include <popops/Zero.hpp>
#include <popops/Loop.hpp>
#include <popops/TopK.hpp>
#include <popops/SortOrder.hpp>
#include<ctime>

// g++ --std=c++11 maxmul_api.cpp -lpoplar -lpopops -lpoputil -lpoplin -o matrixMulApi
using namespace std;
using namespace poplar;
using namespace poplin;
using namespace poplar::program;

program::Sequence check_done(Graph& graph, Tensor& d_done){

    Sequence check_done_prog;

    auto check_done_cs = graph.addComputeSet("check_done_cs");

    auto vtx = graph.addVertex(check_done_cs, "CheckDoneVertex");
    graph.connect(vtx["d_done"], d_done[0]);
    graph.setTileMapping(vtx, 1471);
    check_done_prog.add(Execute(check_done_cs));

    return check_done_prog;
}

void update_step(Graph& graph, Tensor& des, Tensor& val, poplar::program::Sequence& prog){
    auto update_step_cs = graph.addComputeSet("update_step_cs");

    auto vtx = graph.addVertex(update_step_cs, "UpdateStepVertex");
    graph.connect(vtx["des"], des[0]);
    graph.connect(vtx["val"], val[0]);
    graph.setTileMapping(vtx, 1471);

    prog.add(Execute(update_step_cs));
}

void update_scale(Graph& graph, Tensor& des, 
Tensor& val, poplar::program::Sequence& prog, 
int tileIndex, const poplar::DebugContext &debugContext = {}){

    auto update_scale_cs = graph.addComputeSet(debugContext);

    auto vtx = graph.addVertex(update_scale_cs, "UpdateScaleVertex");
    graph.connect(vtx["des"], des[0]);
    graph.connect(vtx["val"], val[0]);
    graph.setTileMapping(vtx, tileIndex);
    prog.add(Execute(update_scale_cs));
}

void update_tensor(Graph& graph, Tensor& t, 
Tensor& val, Tensor& index,
poplar::program::Sequence& prog, 
const poplar::DebugContext &debugContext = {}){

    auto update_tensor_cs = graph.addComputeSet(debugContext);
    auto n = t.dim(0);
    int cnt = 0;
    for(int i = 0; i < n; i += 32){
        auto vtx = graph.addVertex(update_tensor_cs, "UpdateTensorVertex");
        graph.connect(vtx["left"], i);
        graph.connect(vtx["t"], t.slice(i, i+32));
        graph.connect(vtx["val"], val[0]);
        graph.connect(vtx["index"], index[0]);
        graph.setTileMapping(vtx, cnt);
        cnt ++;
    }

    prog.add(Execute(update_tensor_cs));

}

void slice_tensor_with_output(Graph& graph, Tensor& t, 
Tensor& index, Tensor& output, 
Tensor& slice_tensor, poplar::program::Sequence& prog, const poplar::DebugContext &debugContext = {}){
    int cnt = 0;
    auto slice_cs = graph.addComputeSet(debugContext);
    auto n = t.dim(0);
    for(int i = 0; i < n; i += 32){
        auto vtx = graph.addVertex(slice_cs, "SliceTensorVertex");
        graph.connect(vtx["t"], t.slice(i, i+32));
        graph.connect(vtx["index"], index[0]);
        graph.connect(vtx["slice_tensor"], slice_tensor[cnt]);
        graph.setTileMapping(vtx, cnt);
        cnt ++;
    }

    prog.add(Execute(slice_cs));

    auto slice_cs2 = graph.addComputeSet("slice_cs2");
    auto vtx = graph.addVertex(slice_cs2, "SliceTensorVertex2");
    graph.connect(vtx["slice_tensor"], slice_tensor);
    graph.connect(vtx["index"], index[0]);
    graph.connect(vtx["res"], output[0]);
    graph.setTileMapping(vtx, 1471);

    prog.add(Execute(slice_cs2));
}

poplar::Tensor slice_tensor_op(Graph& graph, Tensor& t, 
Tensor& index, Tensor& slice_tensor, 
poplar::program::Sequence& prog, const poplar::DebugContext &debugContext = {}){

    int cnt = 0;
    auto slice_cs = graph.addComputeSet(debugContext);
    auto n = t.dim(0);
    for(int i = 0; i < n; i += 32){
        auto vtx = graph.addVertex(slice_cs, "SliceTensorVertex");
        graph.connect(vtx["t"], t.slice(i, i+32));
        graph.connect(vtx["index"], index[0]);
        graph.connect(vtx["slice_tensor"], slice_tensor[cnt]);
        graph.setTileMapping(vtx, cnt);
        cnt ++;
    }

    prog.add(Execute(slice_cs));

    Tensor res = graph.addVariable(INT, {1}, "res");
    graph.setTileMapping(res, 1471);

    auto slice_cs2 = graph.addComputeSet("slice_cs2");
    auto vtx = graph.addVertex(slice_cs2, "SliceTensorVertex2");
    graph.connect(vtx["slice_tensor"], slice_tensor);
    graph.connect(vtx["index"], index[0]);
    graph.connect(vtx["res"], res[0]);
    graph.setTileMapping(vtx, 1471);
    
    prog.add(Execute(slice_cs2));
    
    return res;
}

program::Sequence step1(Graph& graph, Tensor& d_matrix, Tensor& stepCount){

    int cnt = 0;
    Sequence step1_prog;
    auto n = d_matrix.dim(0);
    auto numTiles = graph.getTarget().getNumTiles()-1;

    Tensor row_min = graph.addVariable(FLOAT, {n}, "row_min");
    for(int i = 0; i < n; i += 2){
        graph.setTileMapping(row_min.slice(i, i+2), cnt);
        cnt ++;
        if(cnt == 1471){
            cnt = 0;
        }
    }

    cnt = 0;
    auto reduce_min = popops::ReduceParams(popops::Operation::MIN);
    std::vector<poplar::ComputeSet> row_min_cs;
    // The minimum value for each row
    reduceWithOutput(graph, d_matrix, row_min, {1}, reduce_min, row_min_cs, "min_each_row_step1");
    for(const auto &cs : row_min_cs){
        step1_prog.add(Execute(cs));
    }

    int block_size = n/6;
    if (block_size % 2) block_size -= 1;

    ComputeSet subtract_row_min_cs = graph.addComputeSet("subtract_row_min_cs_step1");
    for(int i = 0; i < n; i += 2){
        for(int j = i; j < i+2; j ++){
            for(int p = 0; p < 6; p ++){
                int left = p*block_size;
                int right = (p+1)*block_size;
                if(p == 5) right = n;
                auto vtx = graph.addVertex(subtract_row_min_cs, "RowMinSubtract");
                graph.connect(vtx["row"], d_matrix[j].slice(left, right));
                graph.connect(vtx["row_min"], row_min[j]);
                graph.setTileMapping(vtx, cnt);
            }
        }
        cnt ++;
        if(cnt == 1471){
            cnt = 0;
        }
    }

    step1_prog.add(Execute(subtract_row_min_cs));

    std::vector<poplar::ComputeSet> col_min_cs;
    Tensor col_min = reduce(graph, d_matrix, {0}, reduce_min, col_min_cs, "min_each_col_step1");
    for(const auto &cs : col_min_cs){
        step1_prog.add(Execute(cs));
    }

    popops::subInPlace(graph, d_matrix, col_min, step1_prog, "subtract_col_min_cs_step1");


    Tensor one = graph.addVariable(INT, {1}, "one");
    graph.setTileMapping(one, 0);
    graph.setInitialValue(one, ArrayRef<int>{1});
    
    popops::addInPlace(graph, stepCount[0], one, step1_prog);

    return step1_prog;

}

program::Sequence compress_matrix(Graph& graph, Tensor& d_matrix, Tensor& d_zeros, Tensor& zero_count){
    
    auto numTiles = graph.getTarget().getNumTiles()-1;
    auto n = d_matrix.dim(0);
    int block_size = n/6;
    if (block_size % 2) block_size -= 1;

    int cnt = 0;
    ComputeSet compress_cs = graph.addComputeSet("compress_cs");
    Sequence compress_prog;
    for(int i = 0; i < n; i += 2){
        for(int j = i; j < i+2; j ++){
            for(int p = 0; p < 6; p ++){
                int left = p*block_size;
                int right = (p+1)*block_size;
                if(p == 5) right = n;
                auto vtx = graph.addVertex(compress_cs, "CompressVertex");
                graph.connect(vtx["start_index"], left);
                graph.connect(vtx["row"], d_matrix[j].slice(left, right));
                graph.connect(vtx["zero_count"], zero_count[j][p]);
                graph.connect(vtx["d_zeros"], d_zeros[j].slice(p*40, p*40+40));
                graph.setTileMapping(vtx, cnt);
            }
        }
        cnt ++;
        if(cnt == 1471){
            cnt = 0;
        }
    }

    compress_prog.add(Execute(compress_cs));

    return compress_prog; 
}

program::Sequence loopForStar(Graph& graph, Tensor& count, Tensor& d_zero_sorted, Tensor& col_items, Tensor& row_star, Tensor& col_star){

    const auto numTiles = graph.getTarget().getNumTiles();
    auto n  = d_zero_sorted.dim(0);

    Sequence star_prog;
    auto get_col_cs = graph.addComputeSet("get_col_cs");
    
    for(int i = 0; i < n; i ++){
        auto vtx = graph.addVertex(get_col_cs, "GetColVertex");
        graph.connect(vtx["d_zero_sorted"], d_zero_sorted[i]);
        graph.connect(vtx["count"], count[0]);
        graph.connect(vtx["col_items"], col_items[i]);
        graph.setTileMapping(vtx, i%numTiles);
    }
    star_prog.add(Execute(get_col_cs));

    auto star_cs = graph.addComputeSet("star_cs");
    auto vtx = graph.addVertex(star_cs, "StarVertex");
    graph.connect(vtx["row_star"], row_star);
    graph.connect(vtx["col_star"], col_star);
    graph.connect(vtx["col_items"], col_items);

    graph.setTileMapping(vtx, 1471);

    star_prog.add(Execute(star_cs));
    
    return star_prog;
}

program::Sequence step2(Graph& graph, Tensor& row_star, Tensor& col_star, Tensor& d_zeros, Tensor& zero_count, Tensor& total_zero_count, Tensor& stepCount){

    Sequence step2_prog;
    auto numTiles = graph.getTarget().getNumTiles()-1;
    auto n = row_star.dim(0);

    // Reduce sum for the zero_count
    auto reduce_add = popops::ReduceParams(popops::Operation::ADD);
    std::vector<poplar::ComputeSet> sum_zero_count_cs;
    reduceWithOutput(graph, zero_count, total_zero_count, {1}, reduce_add, sum_zero_count_cs, "zero_count_each_row");
    for(const auto &cs : sum_zero_count_cs){
        step2_prog.add(Execute(cs));
    }
    
    Tensor d_zero_sorted = graph.addVariable(INT, {n, 240}, "d_zero_sorted");
    int cnt = 0;
    for(int i = 0; i < n; i += 32){
        graph.setTileMapping(d_zero_sorted.slice(i, i+32), cnt);
        cnt ++;
    }

    d_zero_sorted = popops::sort(graph, d_zeros, {1}, step2_prog, "step2_prog");

    // Get the max of the total_zero_count;
    Tensor max_zero_count = graph.addVariable(INT, {1}, "max_zero_count");
    graph.setTileMapping(max_zero_count, 1471);

    auto reduce_max_param = popops::ReduceParams(popops::Operation::MAX);
    std::vector<poplar::ComputeSet> max_cs;
    // Get the maximum value fo the reduce_max_param
    reduceWithOutput(graph, total_zero_count, max_zero_count, {0}, reduce_max_param, max_cs, "reduce_max_count");
    for(const auto &cs : max_cs){
        step2_prog.add(Execute(cs));
    }

    Tensor count = graph.addVariable(INT, {1}, "count");
    graph.setTileMapping(count, 1471);
    graph.setInitialValue(count, ArrayRef<int>{0});

    Tensor col_items = graph.addVariable(INT, {n}, "col");
    graph.setTileMapping(col_items, 1471);

    // step2_prog.add(PrintTensor("total_zero_count", total_zero_count));
    // step2_prog.add(PrintTensor("max_zero_count", max_zero_count));

    step2_prog.add(  popops::countedForLoop(graph, count, 0, max_zero_count, 1, loopForStar(graph, count, d_zero_sorted, col_items, row_star, col_star))  );

    
    Tensor one = graph.addVariable(INT, {1}, "one");
    graph.setTileMapping(one, 0);
    graph.setInitialValue(one, ArrayRef<int>{1});
    
    popops::addInPlace(graph, stepCount[1], one, step2_prog);

    return step2_prog;
}

program::Sequence step3(Graph& graph, Tensor& col_star, Tensor& col_cover, Tensor& d_n, 
Tensor& d_step, Tensor& d_done, Tensor& stepCount){
    
    Sequence step3_prog;
    auto n = col_star.dim(0);
    auto numTiles = graph.getTarget().getNumTiles();

    ComputeSet update_col_cover_cs = graph.addComputeSet("update_col_cover_cs");

    int cnt = 0;
    int per_tile = 4;
    // col_star  : 4 -1 5 -1 1
    // col_cover : 1 0  1  0 1

    Tensor t_col_cover_sum = graph.addVariable(INT, {n/32}, "t_col_cover_sum");

    for(int i = 0; i < n; i += 32){
        graph.setTileMapping(t_col_cover_sum[cnt], cnt);
        auto vtx = graph.addVertex(update_col_cover_cs, "Step3Vertex");
        graph.connect(vtx["t_col_cover_sum"], t_col_cover_sum[cnt]);
        graph.connect(vtx["col_star"], col_star.slice(i, i+32));
        graph.connect(vtx["col_cover"], col_cover.slice(i, i+32));
        graph.setTileMapping(vtx, cnt);
        cnt ++;
    }

    step3_prog.add(Execute(update_col_cover_cs));

    // step3_prog.add(PrintTensor("t_col_cover_sum", t_col_cover_sum));

    Tensor cover_sum = graph.addVariable(INT, {1}, "cover_sum");
    graph.setTileMapping(cover_sum, 1471);

    auto reduce_add = popops::ReduceParams(popops::Operation::ADD);
    std::vector<poplar::ComputeSet> cover_sum_cs;
    popops::reduceWithOutput(graph, t_col_cover_sum, cover_sum, {0}, reduce_add, cover_sum_cs, "ColAdd_step3");
    for(const auto &cs : cover_sum_cs){
        step3_prog.add(Execute(cs));
    }

    ComputeSet update_step_cs = graph.addComputeSet("update_step_cs");
    auto vtx = graph.addVertex(update_step_cs, "UpdateStep3Vertex");
    graph.connect(vtx["d_n"], d_n[0]);
    graph.connect(vtx["d_step"], d_step[0]);
    graph.connect(vtx["cover_sum"], cover_sum[0]);
    graph.connect(vtx["d_done"], d_done[0]);
    graph.setTileMapping(vtx, 1471);
    step3_prog.add(Execute(update_step_cs));

    
    Tensor one = graph.addVariable(INT, {1}, "one");
    graph.setTileMapping(one, 0);
    graph.setInitialValue(one, ArrayRef<int>{1});
    
    popops::addInPlace(graph, stepCount[2], one, step3_prog);
    return step3_prog;
}

program::Sequence step4b(Graph& graph, Tensor& zero_status, 
Tensor& row_zero,  Tensor& row_cover, 
Tensor& col_cover, Tensor& col_star,
Tensor& row_prime){
    
    Sequence step4b_prog;

    auto n = zero_status.dim(0);
    auto numTiles = graph.getTarget().getNumTiles()-1;
    int cnt = 0;
    ComputeSet cover_uncover_prime_cs = graph.addComputeSet("cover_uncover_prime_cs");

    for(int i = 0; i < n; i += 32){
        auto vtx = graph.addVertex(cover_uncover_prime_cs, "CoverUncoverPrimeVertex");
        graph.connect(vtx["zero_status"], zero_status);
        graph.connect(vtx["zero_status_row"], zero_status.slice(i, i+32));
        graph.connect(vtx["row_cover"], row_cover.slice(i, i+32));
        graph.connect(vtx["row_prime"], row_prime.slice(i, i+32));
        graph.connect(vtx["row_zero"], row_zero.slice(i, i+32));
        graph.connect(vtx["col_cover"], col_cover.slice(i, i+32));
        graph.connect(vtx["col_star"], col_star.slice(i, i+32));
        graph.setTileMapping(vtx, cnt);
        cnt ++;
    }
    
    step4b_prog.add(Execute(cover_uncover_prime_cs));

    return step4b_prog;
}

program::Sequence step4(Graph&graph, 
Tensor& d_zeros, Tensor& zero_count, 
Tensor& row_star, Tensor& col_star, 
Tensor& row_cover, Tensor& col_cover,
Tensor& row_zero, Tensor& zero_status, 
Tensor& row_prime, Tensor& d_step, Tensor& stepCount){
    
    Sequence step4_prog;

    auto n = d_zeros.dim(0);
    auto numTiles = graph.getTarget().getNumTiles()-1;
    ComputeSet find_zero_cs = graph.addComputeSet("find_zero_cs");

    int cnt = 0;
    for(int i = 0; i < n; i += 32){
        for(int j = i; j < i+32; j ++){
            auto vtx = graph.addVertex(find_zero_cs, "FindZeroVertex");
            graph.connect(vtx["d_zeros"], d_zeros[j]);
            graph.connect(vtx["zero_count"], zero_count[j]);
            graph.connect(vtx["row_cover"], row_cover[j]);
            graph.connect(vtx["col_cover"], col_cover);
            graph.connect(vtx["zero_status"], zero_status[j]);
            graph.connect(vtx["row_zero"], row_zero[j]);
            graph.connect(vtx["row_star"], row_star[j]);
            graph.setTileMapping(vtx, cnt);
        }
        cnt ++;
        if(cnt == 1471){
            cnt = 0;
        }
    }
    step4_prog.add(Execute(find_zero_cs));

    Tensor zero_status_max = graph.addVariable(INT, {1}, "zero_status_max");
    graph.setTileMapping(zero_status_max, 1468);

    auto reduce_max_param = popops::ReduceParams(popops::Operation::MAX);
    std::vector<poplar::ComputeSet> max_cs;
    // Get the maximum value fo the reduce_max_param
    popops::reduceWithOutput(graph, zero_status, zero_status_max, {0}, reduce_max_param, max_cs, "reduce_max_count");
    for(const auto &cs : max_cs){
        step4_prog.add(Execute(cs));
    }

    Tensor go_to_step4 = graph.addVariable(INT, {1}, "go_to_step4");
    graph.setTileMapping(go_to_step4, 1471);

    auto update_step_cs = graph.addComputeSet("update_step_cs");
    auto vtx = graph.addVertex(update_step_cs, "StepUpdateVertex");
    graph.connect(vtx["zero_status_max"], zero_status_max[0]);
    graph.connect(vtx["d_step"], d_step[0]);
    graph.connect(vtx["go_to_step4"], go_to_step4[0]);
    graph.setTileMapping(vtx, 1471);
    step4_prog.add(Execute(update_step_cs));

    Sequence step4b_prog = step4b(graph, zero_status, row_zero, row_cover, col_cover, col_star, row_prime); 


    step4_prog.add(If(go_to_step4[0], step4b_prog, Sequence({})));

    Tensor one = graph.addVariable(INT, {1}, "one");
    graph.setTileMapping(one, 0);
    graph.setInitialValue(one, ArrayRef<int>{1});
    
    popops::addInPlace(graph, stepCount[3], one, step4_prog);
    return step4_prog;   
}

program::Sequence check_col_star(Graph& graph, Tensor& col_star, 
Tensor& zero_position_y, Tensor& true_val, 
Tensor& slice_tensor){
    
    Sequence check_col_star_prog;
    Tensor slice_res = slice_tensor_op(graph, col_star, zero_position_y, slice_tensor, check_col_star_prog, "step5_slice_tensor_op");

    Tensor inf = graph.addConstant(INT, {1}, ArrayRef<int>{2147483647}, "inf");
    graph.setTileMapping(inf, 1471);

    popops::neqWithOutput(graph, slice_res, inf, true_val, check_col_star_prog, "step5_slice_tensor_op");

   return check_col_star_prog; 
}

program::Sequence green_increase(Graph& graph, Tensor& row_of_green_at_column, 
Tensor& row_prime, Tensor& col_star,
Tensor& col_position_star, Tensor& last_prime_col,
Tensor& slice_tensor){

    Sequence green_increase_prog;
    Tensor row_position_prime = slice_tensor_op(graph, col_star, col_position_star, slice_tensor, green_increase_prog, "step5_slice_tensor_op");
    Tensor col_position_prime = slice_tensor_op(graph, row_prime, row_position_prime, slice_tensor, green_increase_prog, "step5_slice_tensor_op");

    update_tensor(graph, row_of_green_at_column, row_position_prime, col_position_prime, green_increase_prog, "step5_update_tensor");
    update_scale(graph, col_position_star, col_position_prime, green_increase_prog, 1471, "step5_update_scale");
    update_scale(graph, last_prime_col, col_position_prime, green_increase_prog, 1471, "step5_update_scale");

    return green_increase_prog;
}

// Step5_a to update the row_of_green_at_column variable
program::Sequence step5_a(Graph& graph, Tensor zero_position_x,
Tensor& row_zero, Tensor& row_of_green_at_column, 
Tensor& col_star, Tensor& row_prime,
Tensor& last_prime_col, Tensor& slice_tensor){

    Sequence step5_prog_a;

    // Get the largest col index by using the largest row index through dynamic slice the row_zero
    Tensor zero_position_y = slice_tensor_op(graph, row_zero, zero_position_x, slice_tensor, step5_prog_a, "step5_slice_tensor_op");
    update_scale(graph, last_prime_col, zero_position_y, step5_prog_a, 1471, "step5_update_scale");
    update_tensor(graph, row_of_green_at_column, zero_position_x, zero_position_y, step5_prog_a, "step5_update_tensor");

    Tensor true_val = graph.addVariable(poplar::BOOL, {1}, "true_val");
    graph.setTileMapping(true_val, 1471);
    graph.setInitialValue(true_val, ArrayRef<int>{true});
    step5_prog_a.add(RepeatWhileTrue(check_col_star(graph, col_star, zero_position_y, true_val, slice_tensor), true_val[0], green_increase(graph, row_of_green_at_column, row_prime, col_star, zero_position_y, last_prime_col, slice_tensor)));

    return step5_prog_a;
}

program::Sequence check_col_star_inf(Graph& graph, Tensor& col_star_position, Tensor& true_val){

    Sequence check_col_star_inf_prog;

    Tensor inf = graph.addConstant(INT, {1}, ArrayRef<int>{2147483647}, "inf"); 
    graph.setTileMapping(inf, 1471);

    popops::neqWithOutput(graph, col_star_position, inf, true_val, check_col_star_inf_prog, "check_col_star_inf_prog");
    return check_col_star_inf_prog; 
}

program::Sequence star_operation(Graph& graph, Tensor& row_of_green_at_column, Tensor& row_star, Tensor& col_star, Tensor& col_star_position, Tensor& slice_tensor){

    Sequence star_prog;
    Tensor row_prime_position = slice_tensor_op(graph, row_of_green_at_column, col_star_position, slice_tensor, star_prog, "step5_slice_tensor_op");
    
    Tensor t_col_star_position = graph.addVariable(INT, {1}, "t_col_star_position");
    graph.setTileMapping(t_col_star_position, 1471);
    update_scale(graph, t_col_star_position, col_star_position, star_prog, 1471, "step5_update_scale");

    slice_tensor_with_output(graph, row_star, row_prime_position, col_star_position, slice_tensor,star_prog, "step5_slice_tensor_with_output");

    update_tensor(graph, row_star, t_col_star_position, row_prime_position, star_prog, "step5_update_tensor");
    update_tensor(graph, col_star, row_prime_position, t_col_star_position, star_prog, "step5_update_tensor");

    return star_prog;
}

program::Sequence step5_b(Graph& graph, Tensor& row_star, Tensor& col_star, Tensor& row_of_green_at_column, Tensor& last_prime_col, Tensor& slice_tensor){

    Sequence step5_prog_b;
    Tensor last_prime_row = slice_tensor_op(graph, row_of_green_at_column, last_prime_col, slice_tensor, step5_prog_b, "step5_slice_tensor_op");
    Tensor col_star_position = slice_tensor_op(graph, row_star, last_prime_row, slice_tensor, step5_prog_b, "step5_slice_tensor_op");

    update_tensor(graph, row_star, last_prime_col, last_prime_row, step5_prog_b, "step5_update_tensor");
    update_tensor(graph, col_star, last_prime_row, last_prime_col, step5_prog_b, "step5_update_tensor");

    Tensor true_val = graph.addVariable(poplar::BOOL, {1}, "true_val");
    graph.setTileMapping(true_val, 1471);
    graph.setInitialValue(true_val, ArrayRef<int>{false});

    step5_prog_b.add(RepeatWhileTrue(check_col_star_inf(graph, col_star_position, true_val), true_val[0], star_operation(graph, row_of_green_at_column, row_star, col_star, col_star_position, slice_tensor)));

    return step5_prog_b;
}

program::Sequence step5(Graph& graph, 
Tensor& zero_status, Tensor& row_zero, 
Tensor& row_of_green_at_column, Tensor& row_star, 
Tensor& col_star, Tensor& row_prime, 
Tensor& row_cover, Tensor& col_cover, 
Tensor& slice_tensor, Tensor& d_step, Tensor& stepCount){

    Sequence step5_prog;
    
    auto n = zero_status.dim(0);
    Tensor max_row_index = graph.addVariable(INT, {n/32}, "max_row_index");
    for(int i = 0; i < n/32; i ++){
        graph.setTileMapping(max_row_index[i], i);
    }

    int cnt = 0;
    auto largest_row_index_cs = graph.addComputeSet("largest_row_index_cs");
    for(int i = 0; i < n; i += 32){
        auto vtx = graph.addVertex(largest_row_index_cs, "MaxRowIndexVertex");
        graph.connect(vtx["row_index"], i);
        graph.connect(vtx["zero_status"], zero_status.slice(i, i+32));
        graph.connect(vtx["max_row_index"], max_row_index[cnt]);
        graph.setTileMapping(vtx, cnt);
        cnt ++;
    }
    step5_prog.add(Execute(largest_row_index_cs));

    auto reduce_max_param = popops::ReduceParams(popops::Operation::MAX);
    std::vector<poplar::ComputeSet> max_cs;
    // Get the maximum value fo the reduce_max_param
    Tensor zero_position_x = popops::reduce(graph, max_row_index, {0}, reduce_max_param, max_cs, "largest_row_index");
    for(const auto &cs : max_cs){
        step5_prog.add(Execute(cs));
    }
    
    // step5_prog.add(PrintTensor("max_row_index",max_row_index));
    // step5_prog.add(PrintTensor("zero_position_x", zero_position_x));

    
    Tensor last_prime_col = graph.addVariable(INT, {1}, "last_prime_col");
    graph.setTileMapping(last_prime_col, 1471);

    Sequence step5_a_prog = step5_a(graph, zero_position_x.expand({0}), row_zero, row_of_green_at_column, col_star, row_prime, last_prime_col, slice_tensor);
    Sequence step5_b_prog = step5_b(graph, row_star, col_star, row_of_green_at_column, last_prime_col, slice_tensor);
    
    step5_prog.add(step5_a_prog);
    step5_prog.add(step5_b_prog);

    cnt = 0;
    ComputeSet remove_prime = graph.addComputeSet("remove_prime_cs");
    for(int i = 0; i < n; i += 32){
        auto vtx = graph.addVertex(remove_prime, "ClearVertex");
        graph.connect(vtx["row_prime"], row_prime.slice(i, i+32));
        graph.connect(vtx["row_of_green_at_column"], row_of_green_at_column.slice(i, i+32));
        graph.setTileMapping(vtx, cnt);
        cnt ++;
    }
    step5_prog.add(Execute(remove_prime));


    // go to step 3;
    Tensor three_val = graph.addConstant<int>(poplar::INT, {1}, {3});
    graph.setTileMapping(three_val, 1471);
    update_step(graph, d_step, three_val, step5_prog);

    popops::zero(graph, row_cover, step5_prog, "set_row_cover_0_step5");
    popops::zero(graph, col_cover, step5_prog, "set_col_cover_0_step5");

    Tensor one = graph.addVariable(INT, {1}, "one");
    graph.setTileMapping(one, 0);
    graph.setInitialValue(one, ArrayRef<int>{1});
    
    popops::addInPlace(graph, stepCount[4], one, step5_prog);

    return step5_prog;
}

program::Sequence step6(Graph& graph, Tensor& d_matrix, 
Tensor& row_cover, Tensor& col_cover, 
Tensor& row_min_step6, Tensor& d_zeros, 
Tensor& zero_count, Tensor& d_step, Tensor& count){

    Sequence step6_prog;
    auto numTiles = graph.getTarget().getNumTiles()-1;
    int n = d_matrix.dim(0);
    int block_size = n/6;
    if(block_size % 2) block_size -= 1;
    int cnt = 0;

    auto row_uncover_min_cs = graph.addComputeSet("row_uncover_min_cs");

    for(int i = 0; i < n; i += 2){
        for(int j = i; j < i+2; j ++){
            for(int p = 0; p < 6; p ++){
                int left = p*block_size;
                int right = (p+1)*block_size;
                if(p == 5) right = n;
                auto vtx = graph.addVertex(row_uncover_min_cs, "RowUncoverMinVertex");
                graph.connect(vtx["row"], d_matrix[j].slice(left, right));
                graph.connect(vtx["row_cover"], row_cover[j]);
                graph.connect(vtx["col_cover"], col_cover.slice(left, right));
                graph.connect(vtx["row_min_step6"], row_min_step6[j][p]);
                graph.setTileMapping(vtx, cnt);
            }
        }
        cnt ++;
        if(cnt == 1471){
            cnt = 0;
        }
    }

    step6_prog.add(Execute(row_uncover_min_cs));

    // find the minimum for each row;
    auto reduce_min = popops::ReduceParams(popops::Operation::MIN);
    std::vector<poplar::ComputeSet> all_min_row_cs;
    Tensor all_row_min = reduce(graph, row_min_step6, {1}, reduce_min, all_min_row_cs, "all_min_row_cs");
    for(const auto &cs : all_min_row_cs){
        step6_prog.add(Execute(cs));
    }
    // find the minimum for all;
    // step6_prog.add(PrintTensor("all_row_min", all_row_min));
    std::vector<poplar::ComputeSet> total_min_cs;
    Tensor total_uncover_min = reduce(graph, all_row_min, {0}, reduce_min, total_min_cs, "total_min_cs");
    for(const auto &cs : total_min_cs){
        step6_prog.add(Execute(cs));
    }

    cnt = 0;
    // Each row minus the minimum value
    auto row_minus_min_cs = graph.addComputeSet("row_minus_min_cs");
    for(int i = 0; i < n; i += 2){
        for(int j = i; j < i+2; j ++){
            for(int p = 0; p < 6; p ++){
                int left = p*block_size;
                int right = (p+1)*block_size;
                if(p == 5) right = n;
                auto vtx = graph.addVertex(row_minus_min_cs, "RowMinusMinVertex");
                graph.connect(vtx["row"], d_matrix[j].slice(left, right));
                graph.connect(vtx["row_cover"], row_cover[j]);
                graph.connect(vtx["col_cover"], col_cover.slice(left, right));
                graph.connect(vtx["total_uncover_min"], total_uncover_min);
                graph.setTileMapping(vtx, cnt); 
            }
        }
        cnt ++;
        if(cnt == 1471){
            cnt = 0;
        }
    }
    
    step6_prog.add(Execute(row_minus_min_cs));

    Sequence compress_prog = compress_matrix(graph, d_matrix, d_zeros, zero_count);
    step6_prog.add(compress_prog);

    Tensor four_val = graph.addVariable(INT, {1}, "four_val");
    graph.setTileMapping(four_val, 1471);
    graph.setInitialValue(four_val, ArrayRef<int>{4});

    update_step(graph, d_step, four_val, step6_prog);

    Tensor one = graph.addVariable(INT, {1}, "one");
    graph.setTileMapping(one, 0);
    graph.setInitialValue(one, ArrayRef<int>{1});
    
    popops::addInPlace(graph, count[5], one, step6_prog);

    return step6_prog;
}

int main(int argc, char **argv){
    if(argc != 2){
        std::cout << "Please the correct number of parameters" << std::endl;
        return 0;
    }
    int n = std::atoi(argv[1]);
    // Get the data CPU data;
    vector<float> h_matrix;
    vector<int> h_col_star(n);
    int inf[4096];
    for(int i = 0; i < n; i ++){
        inf[i] = 2147483647;
    }
    string fileName = "/nethome/chengh/NDSD6/NormalDistribution_4096/NormalDistribution20480000.txt";
    ifstream infile(fileName);
    if (infile) {
        string line;
        while (getline(infile, line)) {
            istringstream iss(line);
            string token;
            while (getline(iss, token, ',')) {
                int num = stoi(token);
                h_matrix.push_back(num);
            }
        }
    }
    else {
        cout << "Unable to open file." << endl;
    }
    auto manager = DeviceManager::createDeviceManager();
    // Attempt to attach to a single IPU:
    auto devices = manager.getDevices(poplar::TargetType::IPU, 1);

    std::cout << "Trying to attach to IPU\n";

    auto it = std::find_if(devices.begin(), devices.end(),
                         [](Device &device) { return device.attach(); });

    if (it == devices.end()) {
        std::cerr << "Error attaching to device\n";
        return 1; // EXIT_FAILURE
    }
    auto device = std::move(*it);
    std::cout << "Attached to IPU " << device.getId() << std::endl;

    auto target = device.getTarget();
    // Get the num of tiles in the IPU
    const auto numTiles = target.getNumTiles()-1;

    Graph graph(target);
    popops::addCodelets(graph);
    poplin::addCodelets(graph);
    graph.addCodelets("codelets.cpp");

    Tensor d_matrix = graph.addVariable(FLOAT, {n, n}, "d_matrix");
    Tensor zero_count = graph.addVariable(INT, {n, 6}, "zero_count");
    Tensor row_min_step6 = graph.addVariable(FLOAT, {n, 6}, "row_min_step6");
    Tensor d_zeros = graph.addVariable(INT, {n, 240}, "d_zeros");
    Tensor total_zero_count = graph.addVariable(INT, {n}, "total_zero_count");
    Tensor row_star = graph.addVariable(INT, {n}, "row_star");
    Tensor col_star = graph.addVariable(INT, {n}, "col_star");
    Tensor row_cover = graph.addVariable(INT, {n}, "row_cover");
    Tensor col_cover = graph.addVariable(FLOAT, {n}, "col_cover");
    Tensor zero_status = graph.addVariable(INT, {n}, "zero_status");
    Tensor d_step = graph.addVariable(INT, {1}, "d_step");
    Tensor d_n = graph.addVariable(INT, {1}, "d_n");
    Tensor d_done = graph.addVariable(INT, {1}, "d_done");
    Tensor row_zero = graph.addVariable(INT, {n}, "row_zero");
    Tensor row_prime = graph.addVariable(INT, {n}, "row_prime");
    Tensor row_of_green_at_column = graph.addVariable(INT, {n}, "row_of_green_at_col");

    Tensor count = graph.addVariable(INT, {6}, "count");
    graph.setTileMapping(count, 0);

    Tensor slice_tensor = graph.addVariable(INT, {n/32}, "slice_tensor");

    graph.setTileMapping(slice_tensor, 1471);
    graph.setTileMapping(d_n, 1471);
    graph.setTileMapping(d_done, 1471);
    graph.setTileMapping(d_step, 1471);

    int cnt = 0;
    for(int i = 0; i < n; i += 2){
        graph.setTileMapping(d_matrix.slice(i, i+2), cnt);
        graph.setTileMapping(row_min_step6.slice(i, i+2), cnt); 
        cnt ++;
        if(cnt == 1471){
            cnt = 0;
        }
    }

    cnt = 0;
    for(int i = 0; i < n; i += 32){
        graph.setTileMapping(d_zeros.slice(i, i+32), cnt);
        graph.setTileMapping(zero_count.slice(i, i+32), cnt);
        graph.setTileMapping(total_zero_count.slice(i, i+32), cnt);
        graph.setTileMapping(row_cover.slice(i, i+32), cnt);
        graph.setTileMapping(zero_status.slice(i, i+32), cnt); 
        graph.setTileMapping(row_zero.slice(i, i+32), cnt);
        graph.setTileMapping(row_prime.slice(i, i+32), cnt);
        graph.setTileMapping(row_star.slice(i, i+32), cnt);
        graph.setTileMapping(col_cover.slice(i, i+32), cnt);
        graph.setTileMapping(col_star.slice(i, i+32), cnt);
        graph.setTileMapping(row_of_green_at_column.slice(i, i+32), cnt);
        cnt ++;
        if(cnt == 1471){
            cnt = 0;
        }
    }

    graph.setInitialValue(d_step, ArrayRef<int>{3});
    graph.setInitialValue(row_star, ArrayRef<int>{inf});
    graph.setInitialValue(col_star, ArrayRef<int>{inf});
    graph.setInitialValue(d_n, ArrayRef<int>{n});
    graph.setInitialValue(row_zero, ArrayRef<int>{inf});
    graph.setInitialValue(row_prime, ArrayRef<int>{inf});

    auto matrix_stream = graph.addHostToDeviceFIFO("matrix_stream", FLOAT, n*n);
    auto col_star_stream = graph.addDeviceToHostFIFO("col_star_stream", INT, n); 

    // Write the input to the device
    Sequence write_prog;
    write_prog.add(Copy(matrix_stream, d_matrix));

    Sequence run;

    // Step 1
    Sequence step1_prog = step1(graph, d_matrix, count);

    // compress_matrix
    Sequence compress_prog = compress_matrix(graph, d_matrix, d_zeros, zero_count);

    // Step 2
    Sequence step2_prog = step2(graph, row_star, col_star, d_zeros, zero_count, total_zero_count, count);

    // Step 3
    Sequence step3_prog = step3(graph, col_star, col_cover, d_n, d_step, d_done, count);

    // Step 4
    Sequence step4_prog = step4(graph, d_zeros, zero_count, row_star, col_star, row_cover, col_cover, row_zero, zero_status, row_prime, d_step, count);
    
    Sequence step5_prog = step5(graph, zero_status, row_zero, row_of_green_at_column, row_star, col_star, row_prime, row_cover, col_cover, slice_tensor, d_step, count);

    Sequence step6_prog = step6(graph, d_matrix, row_cover, col_cover, row_min_step6, d_zeros, zero_count, d_step, count);

    std::vector<std::pair<std::int32_t, Program>> cases(4);
    std::pair<std::int32_t, Program> case3(3, step3_prog);
    std::pair<std::int32_t, Program> case4(4, step4_prog);
    std::pair<std::int32_t, Program> case5(5, step5_prog);
    std::pair<std::int32_t, Program> case6(6, step6_prog);

    cases[0] = case3;
    cases[1] = case4;
    cases[2] = case5;
    cases[3] = case6;

    run.add(step1_prog);
    run.add(compress_prog);
    run.add(step2_prog);
    run.add(step3_prog);
     
     

    run.add(
        RepeatWhileFalse(check_done(graph, d_done), d_done[0], Sequence({Switch(d_step[0], cases, "switch_for_HA")}))
    );
    run.add(PrintTensor("count", count));
    Sequence output;
    output.add(Copy(col_star, col_star_stream));

    string poplarGraph = "hungarian_n_v10_2cnt_"+to_string(n)+".poplar";
    std::vector<program::Program> progs{write_prog, run, output};

    try{
        std::cout << "Trying to load from the disk..." << std::endl;
        auto inf = std::ifstream(poplarGraph);
        auto exe = poplar::Executable::deserialize(inf);
        poplar::Engine engine(std::move(exe));
        engine.load(device);
        std::cout << "Load Success" << std::endl;
        engine.connectStream("matrix_stream", h_matrix.data(), h_matrix.data() + h_matrix.size());
        engine.connectStream("col_star_stream", h_col_star.data(), h_col_star.data() + h_col_star.size());
        engine.run(0);
        Engine::TimerTimePoint run_start = engine.getTimeStamp();
        // clock_t before = clock();
        engine.run(1);
        // clock_t after = clock();
        // double elapsed_time = static_cast<double>(after - before) / CLOCKS_PER_SEC;
        // std::cout << "The code took " << elapsed_time << " seconds to execute." << std::endl;
        Engine::TimerTimePoint run_end = engine.getTimeStamp();
        string timing_run = engine.reportTiming(run_start, run_end);
        engine.run(2);
        std::cout << "time to run = " << timing_run << "\n";
    }catch(const std::exception &e){
        std::cout << "Load failed ... Creat the new graph and program..." << std::endl; 
        auto exe = poplar::compileGraph(graph, progs); 
        auto outf = std::ofstream(poplarGraph);
        exe.serialize(outf);
        poplar::Engine engine(std::move(exe));
        engine.load(device);
        engine.connectStream("matrix_stream", h_matrix.data(), h_matrix.data() + h_matrix.size());
        engine.connectStream("col_star_stream", h_col_star.data(), h_col_star.data() + h_col_star.size());
        engine.run(0);
        Engine::TimerTimePoint run_start = engine.getTimeStamp();
        engine.run(1);
        Engine::TimerTimePoint run_end = engine.getTimeStamp();
        string timing_run = engine.reportTiming(run_start, run_end);
        engine.run(2);
        std::cout << "time to run = " << timing_run << "\n";
    }

    int sum = 0;
    for(int i = 0; i < n; i ++){
        int row = h_col_star[i];
        sum += (int)h_matrix[row*n+i];
    }
    std::cout << "sum = " << sum << std::endl;

    return 0;

}
#include <iostream>
#include <cmath>
#include <stdio.h>
#include <string>
#include <fstream>
#include <time.h>
#include <vector>
#include <sstream>

#include <popops/DynamicSlice.hpp>
#include <popops/Sort.hpp>
#include <popops/Reduce.hpp> 
#include <poplar/DeviceManager.hpp>
#include <poplar/Engine.hpp>
#include <poplar/Graph.hpp>
#include <poputil/TileMapping.hpp>
#include <poplin/MatMul.hpp>
#include <popops/ElementWise.hpp>
#include <popops/codelets.hpp>
#include <poplin/codelets.hpp>
#include <popops/Zero.hpp>
#include <popops/Loop.hpp>
#include <popops/TopK.hpp>
#include <popops/SortOrder.hpp>
#include<ctime>

// g++ --std=c++11 maxmul_api.cpp -lpoplar -lpopops -lpoputil -lpoplin -o matrixMulApi
using namespace std;
using namespace poplar;
using namespace poplin;
using namespace poplar::program;

program::Sequence check_done(Graph& graph, Tensor& d_done){

    Sequence check_done_prog;

    auto check_done_cs = graph.addComputeSet("check_done_cs");

    auto vtx = graph.addVertex(check_done_cs, "CheckDoneVertex");
    graph.connect(vtx["d_done"], d_done[0]);
    graph.setTileMapping(vtx, 1471);
    check_done_prog.add(Execute(check_done_cs));

    return check_done_prog;
}

void update_step(Graph& graph, Tensor& des, Tensor& val, poplar::program::Sequence& prog){
    auto update_step_cs = graph.addComputeSet("update_step_cs");

    auto vtx = graph.addVertex(update_step_cs, "UpdateStepVertex");
    graph.connect(vtx["des"], des[0]);
    graph.connect(vtx["val"], val[0]);
    graph.setTileMapping(vtx, 1471);

    prog.add(Execute(update_step_cs));
}

void update_scale(Graph& graph, Tensor& des, 
Tensor& val, poplar::program::Sequence& prog, 
int tileIndex, const poplar::DebugContext &debugContext = {}){

    auto update_scale_cs = graph.addComputeSet(debugContext);

    auto vtx = graph.addVertex(update_scale_cs, "UpdateScaleVertex");
    graph.connect(vtx["des"], des[0]);
    graph.connect(vtx["val"], val[0]);
    graph.setTileMapping(vtx, tileIndex);
    prog.add(Execute(update_scale_cs));
}

void update_tensor(Graph& graph, Tensor& t, 
Tensor& val, Tensor& index,
poplar::program::Sequence& prog, 
const poplar::DebugContext &debugContext = {}){

    auto update_tensor_cs = graph.addComputeSet(debugContext);
    auto n = t.dim(0);
    int cnt = 0;
    for(int i = 0; i < n; i += 32){
        auto vtx = graph.addVertex(update_tensor_cs, "UpdateTensorVertex");
        graph.connect(vtx["left"], i);
        graph.connect(vtx["t"], t.slice(i, i+32));
        graph.connect(vtx["val"], val[0]);
        graph.connect(vtx["index"], index[0]);
        graph.setTileMapping(vtx, cnt);
        cnt ++;
    }

    prog.add(Execute(update_tensor_cs));

}

void slice_tensor_with_output(Graph& graph, Tensor& t, 
Tensor& index, Tensor& output, 
Tensor& slice_tensor, poplar::program::Sequence& prog, const poplar::DebugContext &debugContext = {}){
    int cnt = 0;
    auto slice_cs = graph.addComputeSet(debugContext);
    auto n = t.dim(0);
    for(int i = 0; i < n; i += 32){
        auto vtx = graph.addVertex(slice_cs, "SliceTensorVertex");
        graph.connect(vtx["t"], t.slice(i, i+32));
        graph.connect(vtx["index"], index[0]);
        graph.connect(vtx["slice_tensor"], slice_tensor[cnt]);
        graph.setTileMapping(vtx, cnt);
        cnt ++;
    }

    prog.add(Execute(slice_cs));

    auto slice_cs2 = graph.addComputeSet("slice_cs2");
    auto vtx = graph.addVertex(slice_cs2, "SliceTensorVertex2");
    graph.connect(vtx["slice_tensor"], slice_tensor);
    graph.connect(vtx["index"], index[0]);
    graph.connect(vtx["res"], output[0]);
    graph.setTileMapping(vtx, 1471);

    prog.add(Execute(slice_cs2));
}

poplar::Tensor slice_tensor_op(Graph& graph, Tensor& t, 
Tensor& index, Tensor& slice_tensor, 
poplar::program::Sequence& prog, const poplar::DebugContext &debugContext = {}){

    int cnt = 0;
    auto slice_cs = graph.addComputeSet(debugContext);
    auto n = t.dim(0);
    for(int i = 0; i < n; i += 32){
        auto vtx = graph.addVertex(slice_cs, "SliceTensorVertex");
        graph.connect(vtx["t"], t.slice(i, i+32));
        graph.connect(vtx["index"], index[0]);
        graph.connect(vtx["slice_tensor"], slice_tensor[cnt]);
        graph.setTileMapping(vtx, cnt);
        cnt ++;
    }

    prog.add(Execute(slice_cs));

    Tensor res = graph.addVariable(INT, {1}, "res");
    graph.setTileMapping(res, 1471);

    auto slice_cs2 = graph.addComputeSet("slice_cs2");
    auto vtx = graph.addVertex(slice_cs2, "SliceTensorVertex2");
    graph.connect(vtx["slice_tensor"], slice_tensor);
    graph.connect(vtx["index"], index[0]);
    graph.connect(vtx["res"], res[0]);
    graph.setTileMapping(vtx, 1471);
    
    prog.add(Execute(slice_cs2));
    
    return res;
}

program::Sequence step1(Graph& graph, Tensor& d_matrix, Tensor& stepCount){

    int cnt = 0;
    Sequence step1_prog;
    auto n = d_matrix.dim(0);
    auto numTiles = graph.getTarget().getNumTiles()-1;

    Tensor row_min = graph.addVariable(FLOAT, {n}, "row_min");
    for(int i = 0; i < n; i += 2){
        graph.setTileMapping(row_min.slice(i, i+2), cnt);
        cnt ++;
        if(cnt == 1471){
            cnt = 0;
        }
    }

    cnt = 0;
    auto reduce_min = popops::ReduceParams(popops::Operation::MIN);
    std::vector<poplar::ComputeSet> row_min_cs;
    // The minimum value for each row
    reduceWithOutput(graph, d_matrix, row_min, {1}, reduce_min, row_min_cs, "min_each_row_step1");
    for(const auto &cs : row_min_cs){
        step1_prog.add(Execute(cs));
    }

    int block_size = n/6;
    if (block_size % 2) block_size -= 1;

    ComputeSet subtract_row_min_cs = graph.addComputeSet("subtract_row_min_cs_step1");
    for(int i = 0; i < n; i += 2){
        for(int j = i; j < i+2; j ++){
            for(int p = 0; p < 6; p ++){
                int left = p*block_size;
                int right = (p+1)*block_size;
                if(p == 5) right = n;
                auto vtx = graph.addVertex(subtract_row_min_cs, "RowMinSubtract");
                graph.connect(vtx["row"], d_matrix[j].slice(left, right));
                graph.connect(vtx["row_min"], row_min[j]);
                graph.setTileMapping(vtx, cnt);
            }
        }
        cnt ++;
        if(cnt == 1471){
            cnt = 0;
        }
    }

    step1_prog.add(Execute(subtract_row_min_cs));

    std::vector<poplar::ComputeSet> col_min_cs;
    Tensor col_min = reduce(graph, d_matrix, {0}, reduce_min, col_min_cs, "min_each_col_step1");
    for(const auto &cs : col_min_cs){
        step1_prog.add(Execute(cs));
    }

    popops::subInPlace(graph, d_matrix, col_min, step1_prog, "subtract_col_min_cs_step1");


    Tensor one = graph.addVariable(INT, {1}, "one");
    graph.setTileMapping(one, 0);
    graph.setInitialValue(one, ArrayRef<int>{1});
    
    // popops::addInPlace(graph, stepCount[0], one, step1_prog);

    return step1_prog;

}

program::Sequence compress_matrix(Graph& graph, Tensor& d_matrix, Tensor& d_zeros, Tensor& zero_count){
    
    auto numTiles = graph.getTarget().getNumTiles()-1;
    auto n = d_matrix.dim(0);
    int block_size = n/6;
    if (block_size % 2) block_size -= 1;

    int cnt = 0;
    ComputeSet compress_cs = graph.addComputeSet("compress_cs");
    Sequence compress_prog;
    for(int i = 0; i < n; i += 2){
        for(int j = i; j < i+2; j ++){
            for(int p = 0; p < 6; p ++){
                int left = p*block_size;
                int right = (p+1)*block_size;
                if(p == 5) right = n;
                auto vtx = graph.addVertex(compress_cs, "CompressVertex");
                graph.connect(vtx["start_index"], left);
                graph.connect(vtx["row"], d_matrix[j].slice(left, right));
                graph.connect(vtx["zero_count"], zero_count[j][p]);
                graph.connect(vtx["d_zeros"], d_zeros[j].slice(p*40, p*40+40));
                graph.setTileMapping(vtx, cnt);
            }
        }
        cnt ++;
        if(cnt == 1471){
            cnt = 0;
        }
    }

    compress_prog.add(Execute(compress_cs));

    return compress_prog; 
}

program::Sequence loopForStar(Graph& graph, Tensor& count, Tensor& d_zero_sorted, Tensor& col_items, Tensor& row_star, Tensor& col_star){

    const auto numTiles = graph.getTarget().getNumTiles();
    auto n  = d_zero_sorted.dim(0);

    Sequence star_prog;
    auto get_col_cs = graph.addComputeSet("get_col_cs");
    
    for(int i = 0; i < n; i ++){
        auto vtx = graph.addVertex(get_col_cs, "GetColVertex");
        graph.connect(vtx["d_zero_sorted"], d_zero_sorted[i]);
        graph.connect(vtx["count"], count[0]);
        graph.connect(vtx["col_items"], col_items[i]);
        graph.setTileMapping(vtx, i%numTiles);
    }
    star_prog.add(Execute(get_col_cs));

    auto star_cs = graph.addComputeSet("star_cs");
    auto vtx = graph.addVertex(star_cs, "StarVertex");
    graph.connect(vtx["row_star"], row_star);
    graph.connect(vtx["col_star"], col_star);
    graph.connect(vtx["col_items"], col_items);

    graph.setTileMapping(vtx, 1471);

    star_prog.add(Execute(star_cs));
    
    return star_prog;
}

program::Sequence step2(Graph& graph, Tensor& row_star, Tensor& col_star, Tensor& d_zeros, Tensor& zero_count, Tensor& total_zero_count, Tensor& stepCount){

    Sequence step2_prog;
    auto numTiles = graph.getTarget().getNumTiles()-1;
    auto n = row_star.dim(0);

    // Reduce sum for the zero_count
    auto reduce_add = popops::ReduceParams(popops::Operation::ADD);
    std::vector<poplar::ComputeSet> sum_zero_count_cs;
    reduceWithOutput(graph, zero_count, total_zero_count, {1}, reduce_add, sum_zero_count_cs, "zero_count_each_row");
    for(const auto &cs : sum_zero_count_cs){
        step2_prog.add(Execute(cs));
    }
    
    Tensor d_zero_sorted = graph.addVariable(INT, {n, 240}, "d_zero_sorted");
    int cnt = 0;
    for(int i = 0; i < n; i += 32){
        graph.setTileMapping(d_zero_sorted.slice(i, i+32), cnt);
        cnt ++;
    }

    d_zero_sorted = popops::sort(graph, d_zeros, {1}, step2_prog, "step2_prog");

    // Get the max of the total_zero_count;
    Tensor max_zero_count = graph.addVariable(INT, {1}, "max_zero_count");
    graph.setTileMapping(max_zero_count, 1471);

    auto reduce_max_param = popops::ReduceParams(popops::Operation::MAX);
    std::vector<poplar::ComputeSet> max_cs;
    // Get the maximum value fo the reduce_max_param
    reduceWithOutput(graph, total_zero_count, max_zero_count, {0}, reduce_max_param, max_cs, "reduce_max_count");
    for(const auto &cs : max_cs){
        step2_prog.add(Execute(cs));
    }

    Tensor count = graph.addVariable(INT, {1}, "count");
    graph.setTileMapping(count, 1471);
    graph.setInitialValue(count, ArrayRef<int>{0});

    Tensor col_items = graph.addVariable(INT, {n}, "col");
    graph.setTileMapping(col_items, 1471);

    // step2_prog.add(PrintTensor("total_zero_count", total_zero_count));
    // step2_prog.add(PrintTensor("max_zero_count", max_zero_count));

    step2_prog.add(  popops::countedForLoop(graph, count, 0, max_zero_count, 1, loopForStar(graph, count, d_zero_sorted, col_items, row_star, col_star))  );

    
    Tensor one = graph.addVariable(INT, {1}, "one");
    graph.setTileMapping(one, 0);
    graph.setInitialValue(one, ArrayRef<int>{1});
    
    // popops::addInPlace(graph, stepCount[1], one, step2_prog);

    return step2_prog;
}

program::Sequence step3(Graph& graph, Tensor& col_star, Tensor& col_cover, Tensor& d_n, 
Tensor& d_step, Tensor& d_done, Tensor& stepCount){
    
    Sequence step3_prog;
    auto n = col_star.dim(0);
    auto numTiles = graph.getTarget().getNumTiles();

    ComputeSet update_col_cover_cs = graph.addComputeSet("update_col_cover_cs");

    int cnt = 0;
    int per_tile = 4;
    // col_star  : 4 -1 5 -1 1
    // col_cover : 1 0  1  0 1

    Tensor t_col_cover_sum = graph.addVariable(INT, {n/32}, "t_col_cover_sum");

    for(int i = 0; i < n; i += 32){
        graph.setTileMapping(t_col_cover_sum[cnt], cnt);
        auto vtx = graph.addVertex(update_col_cover_cs, "Step3Vertex");
        graph.connect(vtx["t_col_cover_sum"], t_col_cover_sum[cnt]);
        graph.connect(vtx["col_star"], col_star.slice(i, i+32));
        graph.connect(vtx["col_cover"], col_cover.slice(i, i+32));
        graph.setTileMapping(vtx, cnt);
        cnt ++;
    }

    step3_prog.add(Execute(update_col_cover_cs));

    // step3_prog.add(PrintTensor("t_col_cover_sum", t_col_cover_sum));

    Tensor cover_sum = graph.addVariable(INT, {1}, "cover_sum");
    graph.setTileMapping(cover_sum, 1471);

    auto reduce_add = popops::ReduceParams(popops::Operation::ADD);
    std::vector<poplar::ComputeSet> cover_sum_cs;
    popops::reduceWithOutput(graph, t_col_cover_sum, cover_sum, {0}, reduce_add, cover_sum_cs, "ColAdd_step3");
    for(const auto &cs : cover_sum_cs){
        step3_prog.add(Execute(cs));
    }

    ComputeSet update_step_cs = graph.addComputeSet("update_step_cs");
    auto vtx = graph.addVertex(update_step_cs, "UpdateStep3Vertex");
    graph.connect(vtx["d_n"], d_n[0]);
    graph.connect(vtx["d_step"], d_step[0]);
    graph.connect(vtx["cover_sum"], cover_sum[0]);
    graph.connect(vtx["d_done"], d_done[0]);
    graph.setTileMapping(vtx, 1471);
    step3_prog.add(Execute(update_step_cs));

    
    Tensor one = graph.addVariable(INT, {1}, "one");
    graph.setTileMapping(one, 0);
    graph.setInitialValue(one, ArrayRef<int>{1});
    
    // popops::addInPlace(graph, stepCount[2], one, step3_prog);
    return step3_prog;
}

program::Sequence step4b(Graph& graph, Tensor& zero_status, 
Tensor& row_zero,  Tensor& row_cover, 
Tensor& col_cover, Tensor& col_star,
Tensor& row_prime){
    
    Sequence step4b_prog;

    auto n = zero_status.dim(0);
    auto numTiles = graph.getTarget().getNumTiles()-1;
    int cnt = 0;
    ComputeSet cover_uncover_prime_cs = graph.addComputeSet("cover_uncover_prime_cs");

    for(int i = 0; i < n; i += 32){
        auto vtx = graph.addVertex(cover_uncover_prime_cs, "CoverUncoverPrimeVertex");
        graph.connect(vtx["zero_status"], zero_status);
        graph.connect(vtx["zero_status_row"], zero_status.slice(i, i+32));
        graph.connect(vtx["row_cover"], row_cover.slice(i, i+32));
        graph.connect(vtx["row_prime"], row_prime.slice(i, i+32));
        graph.connect(vtx["row_zero"], row_zero.slice(i, i+32));
        graph.connect(vtx["col_cover"], col_cover.slice(i, i+32));
        graph.connect(vtx["col_star"], col_star.slice(i, i+32));
        graph.setTileMapping(vtx, cnt);
        cnt ++;
    }
    
    step4b_prog.add(Execute(cover_uncover_prime_cs));

    return step4b_prog;
}

program::Sequence step4(Graph&graph, 
Tensor& d_zeros, Tensor& zero_count, 
Tensor& row_star, Tensor& col_star, 
Tensor& row_cover, Tensor& col_cover,
Tensor& row_zero, Tensor& zero_status, 
Tensor& row_prime, Tensor& d_step, Tensor& stepCount){
    
    Sequence step4_prog;

    auto n = d_zeros.dim(0);
    auto numTiles = graph.getTarget().getNumTiles()-1;
    ComputeSet find_zero_cs = graph.addComputeSet("find_zero_cs");

    int cnt = 0;
    for(int i = 0; i < n; i += 32){
        for(int j = i; j < i+32; j ++){
            auto vtx = graph.addVertex(find_zero_cs, "FindZeroVertex");
            graph.connect(vtx["d_zeros"], d_zeros[j]);
            graph.connect(vtx["zero_count"], zero_count[j]);
            graph.connect(vtx["row_cover"], row_cover[j]);
            graph.connect(vtx["col_cover"], col_cover);
            graph.connect(vtx["zero_status"], zero_status[j]);
            graph.connect(vtx["row_zero"], row_zero[j]);
            graph.connect(vtx["row_star"], row_star[j]);
            graph.setTileMapping(vtx, cnt);
        }
        cnt ++;
        if(cnt == 1471){
            cnt = 0;
        }
    }
    step4_prog.add(Execute(find_zero_cs));

    Tensor zero_status_max = graph.addVariable(INT, {1}, "zero_status_max");
    graph.setTileMapping(zero_status_max, 1468);

    auto reduce_max_param = popops::ReduceParams(popops::Operation::MAX);
    std::vector<poplar::ComputeSet> max_cs;
    // Get the maximum value fo the reduce_max_param
    popops::reduceWithOutput(graph, zero_status, zero_status_max, {0}, reduce_max_param, max_cs, "reduce_max_count");
    for(const auto &cs : max_cs){
        step4_prog.add(Execute(cs));
    }

    Tensor go_to_step4 = graph.addVariable(INT, {1}, "go_to_step4");
    graph.setTileMapping(go_to_step4, 1471);

    auto update_step_cs = graph.addComputeSet("update_step_cs");
    auto vtx = graph.addVertex(update_step_cs, "StepUpdateVertex");
    graph.connect(vtx["zero_status_max"], zero_status_max[0]);
    graph.connect(vtx["d_step"], d_step[0]);
    graph.connect(vtx["go_to_step4"], go_to_step4[0]);
    graph.setTileMapping(vtx, 1471);
    step4_prog.add(Execute(update_step_cs));

    Sequence step4b_prog = step4b(graph, zero_status, row_zero, row_cover, col_cover, col_star, row_prime); 


    step4_prog.add(If(go_to_step4[0], step4b_prog, Sequence({})));

    Tensor one = graph.addVariable(INT, {1}, "one");
    graph.setTileMapping(one, 0);
    graph.setInitialValue(one, ArrayRef<int>{1});
    
    // popops::addInPlace(graph, stepCount[3], one, step4_prog);
    return step4_prog;   
}

program::Sequence check_col_star(Graph& graph, Tensor& col_star, 
Tensor& zero_position_y, Tensor& true_val, 
Tensor& slice_tensor){
    
    Sequence check_col_star_prog;
    Tensor slice_res = slice_tensor_op(graph, col_star, zero_position_y, slice_tensor, check_col_star_prog, "step5_slice_tensor_op");

    Tensor inf = graph.addConstant(INT, {1}, ArrayRef<int>{2147483647}, "inf");
    graph.setTileMapping(inf, 1471);

    popops::neqWithOutput(graph, slice_res, inf, true_val, check_col_star_prog, "step5_slice_tensor_op");

   return check_col_star_prog; 
}

program::Sequence green_increase(Graph& graph, Tensor& row_of_green_at_column, 
Tensor& row_prime, Tensor& col_star,
Tensor& col_position_star, Tensor& last_prime_col,
Tensor& slice_tensor){

    Sequence green_increase_prog;
    Tensor row_position_prime = slice_tensor_op(graph, col_star, col_position_star, slice_tensor, green_increase_prog, "step5_slice_tensor_op");
    Tensor col_position_prime = slice_tensor_op(graph, row_prime, row_position_prime, slice_tensor, green_increase_prog, "step5_slice_tensor_op");

    update_tensor(graph, row_of_green_at_column, row_position_prime, col_position_prime, green_increase_prog, "step5_update_tensor");
    update_scale(graph, col_position_star, col_position_prime, green_increase_prog, 1471, "step5_update_scale");
    update_scale(graph, last_prime_col, col_position_prime, green_increase_prog, 1471, "step5_update_scale");

    return green_increase_prog;
}

// Step5_a to update the row_of_green_at_column variable
program::Sequence step5_a(Graph& graph, Tensor zero_position_x,
Tensor& row_zero, Tensor& row_of_green_at_column, 
Tensor& col_star, Tensor& row_prime,
Tensor& last_prime_col, Tensor& slice_tensor){

    Sequence step5_prog_a;

    // Get the largest col index by using the largest row index through dynamic slice the row_zero
    Tensor zero_position_y = slice_tensor_op(graph, row_zero, zero_position_x, slice_tensor, step5_prog_a, "step5_slice_tensor_op");
    update_scale(graph, last_prime_col, zero_position_y, step5_prog_a, 1471, "step5_update_scale");
    update_tensor(graph, row_of_green_at_column, zero_position_x, zero_position_y, step5_prog_a, "step5_update_tensor");

    Tensor true_val = graph.addVariable(poplar::BOOL, {1}, "true_val");
    graph.setTileMapping(true_val, 1471);
    graph.setInitialValue(true_val, ArrayRef<int>{true});
    step5_prog_a.add(RepeatWhileTrue(check_col_star(graph, col_star, zero_position_y, true_val, slice_tensor), true_val[0], green_increase(graph, row_of_green_at_column, row_prime, col_star, zero_position_y, last_prime_col, slice_tensor)));

    return step5_prog_a;
}

program::Sequence check_col_star_inf(Graph& graph, Tensor& col_star_position, Tensor& true_val){

    Sequence check_col_star_inf_prog;

    Tensor inf = graph.addConstant(INT, {1}, ArrayRef<int>{2147483647}, "inf"); 
    graph.setTileMapping(inf, 1471);

    popops::neqWithOutput(graph, col_star_position, inf, true_val, check_col_star_inf_prog, "check_col_star_inf_prog");
    return check_col_star_inf_prog; 
}

program::Sequence star_operation(Graph& graph, Tensor& row_of_green_at_column, Tensor& row_star, Tensor& col_star, Tensor& col_star_position, Tensor& slice_tensor){

    Sequence star_prog;
    Tensor row_prime_position = slice_tensor_op(graph, row_of_green_at_column, col_star_position, slice_tensor, star_prog, "step5_slice_tensor_op");
    
    Tensor t_col_star_position = graph.addVariable(INT, {1}, "t_col_star_position");
    graph.setTileMapping(t_col_star_position, 1471);
    update_scale(graph, t_col_star_position, col_star_position, star_prog, 1471, "step5_update_scale");

    slice_tensor_with_output(graph, row_star, row_prime_position, col_star_position, slice_tensor,star_prog, "step5_slice_tensor_with_output");

    update_tensor(graph, row_star, t_col_star_position, row_prime_position, star_prog, "step5_update_tensor");
    update_tensor(graph, col_star, row_prime_position, t_col_star_position, star_prog, "step5_update_tensor");

    return star_prog;
}

program::Sequence step5_b(Graph& graph, Tensor& row_star, Tensor& col_star, Tensor& row_of_green_at_column, Tensor& last_prime_col, Tensor& slice_tensor){

    Sequence step5_prog_b;
    Tensor last_prime_row = slice_tensor_op(graph, row_of_green_at_column, last_prime_col, slice_tensor, step5_prog_b, "step5_slice_tensor_op");
    Tensor col_star_position = slice_tensor_op(graph, row_star, last_prime_row, slice_tensor, step5_prog_b, "step5_slice_tensor_op");

    update_tensor(graph, row_star, last_prime_col, last_prime_row, step5_prog_b, "step5_update_tensor");
    update_tensor(graph, col_star, last_prime_row, last_prime_col, step5_prog_b, "step5_update_tensor");

    Tensor true_val = graph.addVariable(poplar::BOOL, {1}, "true_val");
    graph.setTileMapping(true_val, 1471);
    graph.setInitialValue(true_val, ArrayRef<int>{false});

    step5_prog_b.add(RepeatWhileTrue(check_col_star_inf(graph, col_star_position, true_val), true_val[0], star_operation(graph, row_of_green_at_column, row_star, col_star, col_star_position, slice_tensor)));

    return step5_prog_b;
}

program::Sequence step5(Graph& graph, 
Tensor& zero_status, Tensor& row_zero, 
Tensor& row_of_green_at_column, Tensor& row_star, 
Tensor& col_star, Tensor& row_prime, 
Tensor& row_cover, Tensor& col_cover, 
Tensor& slice_tensor, Tensor& d_step, Tensor& stepCount){

    Sequence step5_prog;
    
    auto n = zero_status.dim(0);
    Tensor max_row_index = graph.addVariable(INT, {n/32}, "max_row_index");
    for(int i = 0; i < n/32; i ++){
        graph.setTileMapping(max_row_index[i], i);
    }

    int cnt = 0;
    auto largest_row_index_cs = graph.addComputeSet("largest_row_index_cs");
    for(int i = 0; i < n; i += 32){
        auto vtx = graph.addVertex(largest_row_index_cs, "MaxRowIndexVertex");
        graph.connect(vtx["row_index"], i);
        graph.connect(vtx["zero_status"], zero_status.slice(i, i+32));
        graph.connect(vtx["max_row_index"], max_row_index[cnt]);
        graph.setTileMapping(vtx, cnt);
        cnt ++;
    }
    step5_prog.add(Execute(largest_row_index_cs));

    auto reduce_max_param = popops::ReduceParams(popops::Operation::MAX);
    std::vector<poplar::ComputeSet> max_cs;
    // Get the maximum value fo the reduce_max_param
    Tensor zero_position_x = popops::reduce(graph, max_row_index, {0}, reduce_max_param, max_cs, "largest_row_index");
    for(const auto &cs : max_cs){
        step5_prog.add(Execute(cs));
    }
    
    // step5_prog.add(PrintTensor("max_row_index",max_row_index));
    // step5_prog.add(PrintTensor("zero_position_x", zero_position_x));

    
    Tensor last_prime_col = graph.addVariable(INT, {1}, "last_prime_col");
    graph.setTileMapping(last_prime_col, 1471);

    Sequence step5_a_prog = step5_a(graph, zero_position_x.expand({0}), row_zero, row_of_green_at_column, col_star, row_prime, last_prime_col, slice_tensor);
    Sequence step5_b_prog = step5_b(graph, row_star, col_star, row_of_green_at_column, last_prime_col, slice_tensor);
    
    step5_prog.add(step5_a_prog);
    step5_prog.add(step5_b_prog);

    cnt = 0;
    ComputeSet remove_prime = graph.addComputeSet("remove_prime_cs");
    for(int i = 0; i < n; i += 32){
        auto vtx = graph.addVertex(remove_prime, "ClearVertex");
        graph.connect(vtx["row_prime"], row_prime.slice(i, i+32));
        graph.connect(vtx["row_of_green_at_column"], row_of_green_at_column.slice(i, i+32));
        graph.setTileMapping(vtx, cnt);
        cnt ++;
    }
    step5_prog.add(Execute(remove_prime));


    // go to step 3;
    Tensor three_val = graph.addConstant<int>(poplar::INT, {1}, {3});
    graph.setTileMapping(three_val, 1471);
    update_step(graph, d_step, three_val, step5_prog);

    popops::zero(graph, row_cover, step5_prog, "set_row_cover_0_step5");
    popops::zero(graph, col_cover, step5_prog, "set_col_cover_0_step5");

    Tensor one = graph.addVariable(INT, {1}, "one");
    graph.setTileMapping(one, 0);
    graph.setInitialValue(one, ArrayRef<int>{1});
    
    // popops::addInPlace(graph, stepCount[4], one, step5_prog);

    return step5_prog;
}

program::Sequence step6(Graph& graph, Tensor& d_matrix, 
Tensor& row_cover, Tensor& col_cover, 
Tensor& row_min_step6, Tensor& d_zeros, 
Tensor& zero_count, Tensor& d_step, Tensor& count){

    Sequence step6_prog;
    auto numTiles = graph.getTarget().getNumTiles()-1;
    int n = d_matrix.dim(0);
    int block_size = n/6;
    if(block_size % 2) block_size -= 1;
    int cnt = 0;

    auto row_uncover_min_cs = graph.addComputeSet("row_uncover_min_cs");

    for(int i = 0; i < n; i += 2){
        for(int j = i; j < i+2; j ++){
            for(int p = 0; p < 6; p ++){
                int left = p*block_size;
                int right = (p+1)*block_size;
                if(p == 5) right = n;
                auto vtx = graph.addVertex(row_uncover_min_cs, "RowUncoverMinVertex");
                graph.connect(vtx["row"], d_matrix[j].slice(left, right));
                graph.connect(vtx["row_cover"], row_cover[j]);
                graph.connect(vtx["col_cover"], col_cover.slice(left, right));
                graph.connect(vtx["row_min_step6"], row_min_step6[j][p]);
                graph.setTileMapping(vtx, cnt);
            }
        }
        cnt ++;
        if(cnt == 1471){
            cnt = 0;
        }
    }

    step6_prog.add(Execute(row_uncover_min_cs));

    // find the minimum for each row;
    auto reduce_min = popops::ReduceParams(popops::Operation::MIN);
    std::vector<poplar::ComputeSet> all_min_row_cs;
    Tensor all_row_min = reduce(graph, row_min_step6, {1}, reduce_min, all_min_row_cs, "all_min_row_cs");
    for(const auto &cs : all_min_row_cs){
        step6_prog.add(Execute(cs));
    }
    // find the minimum for all;
    // step6_prog.add(PrintTensor("all_row_min", all_row_min));
    std::vector<poplar::ComputeSet> total_min_cs;
    Tensor total_uncover_min = reduce(graph, all_row_min, {0}, reduce_min, total_min_cs, "total_min_cs");
    for(const auto &cs : total_min_cs){
        step6_prog.add(Execute(cs));
    }

    cnt = 0;
    // Each row minus the minimum value
    auto row_minus_min_cs = graph.addComputeSet("row_minus_min_cs");
    for(int i = 0; i < n; i += 2){
        for(int j = i; j < i+2; j ++){
            for(int p = 0; p < 6; p ++){
                int left = p*block_size;
                int right = (p+1)*block_size;
                if(p == 5) right = n;
                auto vtx = graph.addVertex(row_minus_min_cs, "RowMinusMinVertex");
                graph.connect(vtx["row"], d_matrix[j].slice(left, right));
                graph.connect(vtx["row_cover"], row_cover[j]);
                graph.connect(vtx["col_cover"], col_cover.slice(left, right));
                graph.connect(vtx["total_uncover_min"], total_uncover_min);
                graph.setTileMapping(vtx, cnt); 
            }
        }
        cnt ++;
        if(cnt == 1471){
            cnt = 0;
        }
    }
    
    step6_prog.add(Execute(row_minus_min_cs));

    Sequence compress_prog = compress_matrix(graph, d_matrix, d_zeros, zero_count);
    step6_prog.add(compress_prog);

    Tensor four_val = graph.addVariable(INT, {1}, "four_val");
    graph.setTileMapping(four_val, 1471);
    graph.setInitialValue(four_val, ArrayRef<int>{4});

    update_step(graph, d_step, four_val, step6_prog);

    Tensor one = graph.addVariable(INT, {1}, "one");
    graph.setTileMapping(one, 0);
    graph.setInitialValue(one, ArrayRef<int>{1});
    
    popops::addInPlace(graph, count[5], one, step6_prog);

    return step6_prog;
}

int main(int argc, char **argv){
    if(argc != 3){
        std::cout << "Please the correct number of parameters" << std::endl;
        return 0;
    }
    int n = std::atoi(argv[1]);
    // Get the data CPU data;
    vector<float> h_matrix;
    vector<int> h_col_star(n);
    vector<int> inf(n);
    for(int i = 0; i < n; i ++){
        inf[i] = 2147483647;
    } 
    string fileName = argv[2];
    std::cout << "fileName = " << fileName << std::endl;
    ifstream infile(fileName);
    if (infile) {
        string line;
        while (getline(infile, line)) {
            istringstream iss(line);
            string token;
            while (getline(iss, token, ',')) {
                int num = stoi(token);
                h_matrix.push_back(num);
            }
        }
    }
    else {
        cout << "Unable to open file." << endl;
    }
    auto manager = DeviceManager::createDeviceManager();
    // Attempt to attach to a single IPU:
    auto devices = manager.getDevices(poplar::TargetType::IPU, 1);

    std::cout << "Trying to attach to IPU\n";

    auto it = std::find_if(devices.begin(), devices.end(),
                         [](Device &device) { return device.attach(); });

    if (it == devices.end()) {
        std::cerr << "Error attaching to device\n";
        return 1; // EXIT_FAILURE
    }
    auto device = std::move(*it);
    std::cout << "Attached to IPU " << device.getId() << std::endl;

    auto target = device.getTarget();
    // Get the num of tiles in the IPU
    const auto numTiles = target.getNumTiles()-1;

    Graph graph(target);
    popops::addCodelets(graph);
    poplin::addCodelets(graph);
    graph.addCodelets("codelets.cpp");

    Tensor d_matrix = graph.addVariable(FLOAT, {n, n}, "d_matrix");
    Tensor zero_count = graph.addVariable(INT, {n, 6}, "zero_count");
    Tensor row_min_step6 = graph.addVariable(FLOAT, {n, 6}, "row_min_step6");
    Tensor d_zeros = graph.addVariable(INT, {n, 240}, "d_zeros");
    Tensor total_zero_count = graph.addVariable(INT, {n}, "total_zero_count");
    Tensor row_star = graph.addVariable(INT, {n}, "row_star");
    Tensor col_star = graph.addVariable(INT, {n}, "col_star");
    Tensor row_cover = graph.addVariable(INT, {n}, "row_cover");
    Tensor col_cover = graph.addVariable(FLOAT, {n}, "col_cover");
    Tensor zero_status = graph.addVariable(INT, {n}, "zero_status");
    Tensor d_step = graph.addVariable(INT, {1}, "d_step");
    Tensor d_n = graph.addVariable(INT, {1}, "d_n");
    Tensor d_done = graph.addVariable(INT, {1}, "d_done");
    Tensor row_zero = graph.addVariable(INT, {n}, "row_zero");
    Tensor row_prime = graph.addVariable(INT, {n}, "row_prime");
    Tensor row_of_green_at_column = graph.addVariable(INT, {n}, "row_of_green_at_col");

    Tensor count = graph.addVariable(INT, {6}, "count");
    graph.setTileMapping(count, 0);

    Tensor slice_tensor = graph.addVariable(INT, {n/32}, "slice_tensor");

    graph.setTileMapping(slice_tensor, 1471);
    graph.setTileMapping(d_n, 1471);
    graph.setTileMapping(d_done, 1471);
    graph.setTileMapping(d_step, 1471);

    int cnt = 0;
    for(int i = 0; i < n; i += 2){
        graph.setTileMapping(d_matrix.slice(i, i+2), cnt);
        graph.setTileMapping(row_min_step6.slice(i, i+2), cnt); 
        cnt ++;
        if(cnt == 1471){
            cnt = 0;
        }
    }

    cnt = 0;
    for(int i = 0; i < n; i += 32){
        graph.setTileMapping(d_zeros.slice(i, i+32), cnt);
        graph.setTileMapping(zero_count.slice(i, i+32), cnt);
        graph.setTileMapping(total_zero_count.slice(i, i+32), cnt);
        graph.setTileMapping(row_cover.slice(i, i+32), cnt);
        graph.setTileMapping(zero_status.slice(i, i+32), cnt); 
        graph.setTileMapping(row_zero.slice(i, i+32), cnt);
        graph.setTileMapping(row_prime.slice(i, i+32), cnt);
        graph.setTileMapping(row_star.slice(i, i+32), cnt);
        graph.setTileMapping(col_cover.slice(i, i+32), cnt);
        graph.setTileMapping(col_star.slice(i, i+32), cnt);
        graph.setTileMapping(row_of_green_at_column.slice(i, i+32), cnt);
        cnt ++;
        if(cnt == 1471){
            cnt = 0;
        }
    }

    graph.setInitialValue(d_step, ArrayRef<int>{3});
    graph.setInitialValue(row_star, ArrayRef<int>{inf});
    graph.setInitialValue(col_star, ArrayRef<int>{inf});
    graph.setInitialValue(d_n, ArrayRef<int>{n});
    graph.setInitialValue(row_zero, ArrayRef<int>{inf});
    graph.setInitialValue(row_prime, ArrayRef<int>{inf});

    auto matrix_stream = graph.addHostToDeviceFIFO("matrix_stream", FLOAT, n*n);
    auto col_star_stream = graph.addDeviceToHostFIFO("col_star_stream", INT, n); 

    // Write the input to the device
    Sequence write_prog;
    write_prog.add(Copy(matrix_stream, d_matrix));

    Sequence run;

    // Step 1
    Sequence step1_prog = step1(graph, d_matrix, count);

    // compress_matrix
    Sequence compress_prog = compress_matrix(graph, d_matrix, d_zeros, zero_count);

    // Step 2
    Sequence step2_prog = step2(graph, row_star, col_star, d_zeros, zero_count, total_zero_count, count);

    // Step 3
    Sequence step3_prog = step3(graph, col_star, col_cover, d_n, d_step, d_done, count);

    // Step 4
    Sequence step4_prog = step4(graph, d_zeros, zero_count, row_star, col_star, row_cover, col_cover, row_zero, zero_status, row_prime, d_step, count);
    
    Sequence step5_prog = step5(graph, zero_status, row_zero, row_of_green_at_column, row_star, col_star, row_prime, row_cover, col_cover, slice_tensor, d_step, count);

    Sequence step6_prog = step6(graph, d_matrix, row_cover, col_cover, row_min_step6, d_zeros, zero_count, d_step, count);

    std::vector<std::pair<std::int32_t, Program>> cases(4);
    std::pair<std::int32_t, Program> case3(3, step3_prog);
    std::pair<std::int32_t, Program> case4(4, step4_prog);
    std::pair<std::int32_t, Program> case5(5, step5_prog);
    std::pair<std::int32_t, Program> case6(6, step6_prog);

    cases[0] = case3;
    cases[1] = case4;
    cases[2] = case5;
    cases[3] = case6;

    run.add(step1_prog);
    run.add(compress_prog);
    run.add(step2_prog);
    run.add(step3_prog);
     
     

    run.add(
        RepeatWhileFalse(check_done(graph, d_done), d_done[0], Sequence({Switch(d_step[0], cases, "switch_for_HA")}))
    );
    // run.add(PrintTensor("count", count));
    Sequence output;
    output.add(Copy(col_star, col_star_stream));

    string poplarGraph = "hungarian_n_v10_2cnt_"+to_string(n)+".poplar";
    std::vector<program::Program> progs{write_prog, run, output};

    try{
        std::cout << "Trying to load from the disk..." << std::endl;
        auto inf = std::ifstream(poplarGraph);
        auto exe = poplar::Executable::deserialize(inf);
        poplar::Engine engine(std::move(exe));
        engine.load(device);
        std::cout << "Load Success" << std::endl;
        engine.connectStream("matrix_stream", h_matrix.data(), h_matrix.data() + h_matrix.size());
        engine.connectStream("col_star_stream", h_col_star.data(), h_col_star.data() + h_col_star.size());
        engine.run(0);
        Engine::TimerTimePoint run_start = engine.getTimeStamp();
        // clock_t before = clock();
        engine.run(1);
        // clock_t after = clock();
        // double elapsed_time = static_cast<double>(after - before) / CLOCKS_PER_SEC;
        // std::cout << "The code took " << elapsed_time << " seconds to execute." << std::endl;
        Engine::TimerTimePoint run_end = engine.getTimeStamp();
        string timing_run = engine.reportTiming(run_start, run_end);
        engine.run(2);
        std::cout << "time to run = " << timing_run << "\n";
    }catch(const std::exception &e){
        std::cout << "Load failed ... Creat the new graph and program..." << std::endl; 
        auto exe = poplar::compileGraph(graph, progs); 
        auto outf = std::ofstream(poplarGraph);
        exe.serialize(outf);
        poplar::Engine engine(std::move(exe));
        engine.load(device);
        engine.connectStream("matrix_stream", h_matrix.data(), h_matrix.data() + h_matrix.size());
        engine.connectStream("col_star_stream", h_col_star.data(), h_col_star.data() + h_col_star.size());
        engine.run(0);
        Engine::TimerTimePoint run_start = engine.getTimeStamp();
        engine.run(1);
        Engine::TimerTimePoint run_end = engine.getTimeStamp();
        string timing_run = engine.reportTiming(run_start, run_end);
        engine.run(2);
        std::cout << "time to run = " << timing_run << "\n";
    }

    int sum = 0;
    for(int i = 0; i < n; i ++){
        int row = h_col_star[i];
        sum += (int)h_matrix[row*n+i];
    }
    std::cout << "sum = " << sum << std::endl;

    return 0;

}