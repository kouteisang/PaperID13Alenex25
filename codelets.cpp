#include <poplar/Vertex.hpp>
#include <print.h>
#include <poplar/VectorListTypes.hpp>
#include <ipu_builtins.h>
#include <float.h>
using namespace poplar;

class RowMinSubtract : public Vertex{

public:
    InOut<Vector<float, VectorLayout::SPAN, 8>> row;
    Input<float> row_min;

    void compute(){
        
        const int n = row.size();
        auto n2 = n/2;

        const float2 t_row_min = {row_min, row_min};
        float2* row2 = (float2*) &row[0];
    
        for(unsigned i = 0; i < n2; i ++){
            row2[i] -= t_row_min;
        }
        if(n%2 == 1){
            row[n-1] -= row_min;
        }
    }
};


class CompressVertex : public Vertex{

public:

    Input<int> start_index;
    Input<Vector<float, VectorLayout::SPAN, 8>> row;
    Output<int> zero_count;
    Output<Vector<int>> d_zeros;

    void compute(){
        
        // d_zeros[0] = 2147483647;
        // d_zeros[1] = 2147483647;
        // d_zeros[2] = 2147483647;
        // d_zeros[3] = 2147483647;
        // d_zeros[4] = 2147483647;

        for(int i = 0; i < 40; i ++){
            d_zeros[i] = 2147483647; 
        }
          
        const int n  = row.size();
        int t_start_index = *start_index;
        int cnt = 0;

        float2* row2 = (float2*) &row[0];
        auto n2 = n/2;

        #pragma clang loop unroll_count(3) 
        for(int i = 0; i < n2; i ++){
            float2 val = row2[i];
            if(val[0] == 0.f){
                d_zeros[cnt] = t_start_index+2*i;
                cnt ++;
            }

            if(val[1] == 0.f){
                d_zeros[cnt] = t_start_index+2*i+1;
                cnt ++;
            }
     
        }
        if(n%2 && row[n-1] == 0.f){
            d_zeros[cnt] = n-1;
            cnt ++;
        }
        *zero_count = cnt;
        return ;
    }
};


class GetColVertex : public Vertex{
public:
    Input<Vector<int>> d_zero_sorted;
    Input<int> count;
    Output<int> col_items;
    
    void compute(){
        *col_items = d_zero_sorted[count];
        return;
    }
};


class StarVertex : public Vertex{
public:
    InOut<Vector<int>> row_star;
    InOut<Vector<int>> col_star;
    Input<Vector<int>> col_items;

    void compute(){
        auto n = col_items.size();

        #pragma clang loop unroll_count(3)
        for(int i = 0; i < n; i ++){
            int col_num = col_items[i];
            if(col_num == 2147483647) continue;
            if(row_star[i] == 2147483647 && col_star[col_num] == 2147483647){
                row_star[i] = col_num;
                col_star[col_num] = i;
            }
        }
      
        return ;
    }
};


class Step3Vertex : public Vertex{
public:
    Input<Vector<int>> col_star;
    Output<Vector<float>> col_cover;
    Output<int> t_col_cover_sum;

    void compute(){
        int t_sum = 0;
        const auto n = col_star.size();
        for(int i = 0; i < n; i ++){
            if(col_star[i] != 2147483647){
                col_cover[i] = 1.0f;
                t_sum ++;
            }else{
                col_cover[i] = 0.0f;
            }
        }
        *t_col_cover_sum = t_sum;
        return ;
    }
};

class UpdateStep3Vertex : public Vertex{
public:
    Input<int> d_n;
    Input<int> cover_sum;
    InOut<int> d_step;
    InOut<int> d_done;

    void compute(){
        if(d_n == cover_sum){
            *d_done = true;
        }else{
            *d_done = false;
            *d_step = 4;
        }
        return ;
    }
};

class FindZeroVertex : public Vertex{
public:
    Input<Vector<int>> d_zeros;
    Input<Vector<int>> zero_count;
    Input<int> row_cover;
    Input<Vector<float>> col_cover;
    Input<int> row_star;
    Output<int> zero_status;
    Output<int> row_zero;

    void compute(){
        if(row_cover == 1){
            *zero_status = -1;
            *row_zero = 2147483647;
            return;
        }
        int n = d_zeros.size();
        for(int i = 0; i < 6; i ++){
            int left = i*40;
            int right = i*40 + zero_count[i];
            // printf("%d---%d", left, right);
            for(int j = left; j < right; j ++){
                int index = d_zeros[j];
                if(col_cover[index] == 0.f){
                    if(row_star != 2147483647){
                        *zero_status = 0;
                        *row_zero = index;
                        return;
                    }else if(row_star == 2147483647){
                        *zero_status = 1;
                        *row_zero = index;
                        return ;
                    }
                }
            }
        }
        *zero_status = -1;
        *row_zero = 2147483647;  
        return ; 
    }
};


class RowUncoverMinVertex : public Vertex{
public:
    Input<Vector<float, VectorLayout::SPAN, 8>> row;
    Input<int> row_cover;
    Input<Vector<float, VectorLayout::SPAN, 8>> col_cover;
    Output<float> row_min_step6;

     void compute(){
        auto n = row.size();
        if(row_cover == 1){
            *row_min_step6 = FLT_MAX;
            return ; 
        }

        const float2 flt_max_v2 = {FLT_MAX, FLT_MAX}; 
        float2 t_min = flt_max_v2;
        float2* row2 = (float2*) &row[0];
        float2* col_cover2 = (float2*) &col_cover[0];
        unsigned n2 = n / 2;

        for(unsigned i = 0; i < n2; i ++){
            float2 values = row2[i];
            values += col_cover2[i] * flt_max_v2;
            t_min = __builtin_ipu_min(values, t_min);
        }
        *row_min_step6 = __builtin_ipu_min(t_min[0], t_min[1]);
        
        // Handle possible last value
        if ((n % 2) && (col_cover[n-1] == 0.f) && (row[n-1] < *row_min_step6)) {
            *row_min_step6 = row[n-1];
        }
    }
};



class UpdateStepVertex : public Vertex{
public:
    Output<int> des;
    Input<int> val;

    void compute(){
        *des = val;
        return ;
    }
};

class StepUpdateVertex : public Vertex{
public:
    Input<int> zero_status_max;
    Output<int> d_step;
    Output<int> go_to_step4;

    void compute(){
        if(zero_status_max == -1){
            *d_step = 6;
            *go_to_step4 = 0;
        }else if(zero_status_max == 0){
            *d_step = 4;
            *go_to_step4 = 1;
        }else if(zero_status_max == 1){
            *d_step = 5;
            *go_to_step4 = 0;
        }
        return ;
    }
};

class CoverUncoverPrimeVertex : public MultiVertex{

public:
    Input<Vector<int>> zero_status;
    Input<Vector<int>> zero_status_row;
    InOut<Vector<int>> row_cover;
    InOut<Vector<int>> row_prime;
    InOut<Vector<int>> row_zero;
    InOut<Vector<float>> col_cover;
    InOut<Vector<int>> col_star;

    void compute(unsigned workId){

        for(int i = workId; i < 32; i += 6){
            if(zero_status_row[i] == 0){
                row_cover[i] = 1;
                row_prime[i] = row_zero[i];
            }
            if(col_star[i] != 2147483647 && zero_status[col_star[i]] == 0){
                col_cover[i] = 0.f;
            }
        }
        return ;
    }

};


class UpdateScaleVertex : public Vertex{

public:
    Output<int> des;
    Input<int> val;

    void compute(){
        *des = val;
        return ;
    }
};


class UpdateTensorVertex : public Vertex{

public:
    Input<int> left;
    InOut<Vector<int>> t;
    Input<int> val;
    Input<int> index;

    void compute(){
        if(index >= left && index < left+32){
            t[index-left] = val;
        }
        return ;
    }

};

// class CheckColStarVertex : public Vertex{

// public:
//     Input<Vector<int>> col_star;
//     Input<int> zero_position_y;
//     Output<bool> true_val;

//     void compute(){
//         if(col_star[zero_position_y] == 2147483647){
//             *true_val = false;
//         }else if(col_star[zero_position_y] != 2147483647){
//             *true_val = true;
//         }
//         return ;
//     }

// };


class SliceTensorVertex : public Vertex{

public:
    Input<Vector<int>> t;
    Input<int> index;
    Output<int> slice_tensor;

    void compute(){
        *slice_tensor = t[index % 32];
        return ;
    }

};

class SliceTensorVertex2 : public Vertex{

public:
    Input<Vector<int>> slice_tensor;
    Input<int> index;
    Output<int> res;

    void compute(){
        *res = slice_tensor[index / 32];
        return ;
    }    
};

class CheckColStarInfVertex : public Vertex{

public:
    Input<int> col_star_position;
    Output<bool> true_val;

    void compute(){
        if(col_star_position == 2147483647){
            *true_val = false;
        }else{
            *true_val = true;
        }
        return ;
    }

};

class SliceTensorWithOutputVertex : public Vertex{

public:
    Input<Vector<int>> t;
    Input<int> index;
    Output<int> output;

    void compute(){
        *output = t[index];
        return ;
    }
};


class ClearVertex : public MultiVertex {
public:
    InOut<Vector<int>> row_prime;
    InOut<Vector<int>> row_of_green_at_column;

    void compute(unsigned workId){
        const auto n = row_prime.size();
        for(std::size_t i = workId; i < n; i += MultiVertex::numWorkers()) {
            row_prime[i] = 2147483647;
            row_of_green_at_column[i] = 2147483647; 
        } 
    }
};


class CheckDoneVertex : public Vertex{

public:
    InOut<int> d_done;

    void compute(){
        if(d_done == 1) {
            *d_done = 1;
        }else{
            *d_done = 0;
        }
        return ;
    }
};



class RowMinusMinVertex : public Vertex{
public:
    InOut<Vector<float, VectorLayout::SPAN, 8>> row;
    Input<int> row_cover;
    Input<Vector<float, VectorLayout::SPAN, 8>> col_cover;
    Input<float> total_uncover_min;

    void compute(){
        int n = row.size();
        float min = *total_uncover_min;

        float2 min2 = {min, min};
        float2* row2 = (float2*) &row[0];
        float2* col_cover2 = (float2*) &col_cover[0];
        rptsize_t n2 = n / 2;

        
        if (row_cover == 0) {
            // #pragma clang loop unroll_count(3) 
            for(unsigned i = 0; i < n2; i ++) {
                row2[i] -= (1.f - col_cover2[i]) * min2;
            }

            if (n % 2) {
                row[n-1] -= col_cover[n-1] * min;
            }

        } else {
            // #pragma clang loop unroll_count(3) 
            for(unsigned  i = 0; i < n2; i ++) {
                row2[i] += col_cover2[i] * min2;
            }

            if (n % 2) {
                row[n-1] += col_cover[n-1] * min;
            }

        } 
        
        return ;
    }
};


class MaxRowIndexVertex : public Vertex{
public:
    Input<int> row_index;
    Input<Vector<int>> zero_status;
    Output<int> max_row_index;

    void compute(){
        int tmp_row = -1;
        int t_row_index = *row_index;
        auto n = zero_status.size();
        for(int i = 0; i < n; i ++){
            if(zero_status[i] == 1){
                tmp_row = t_row_index+i;
                break;
            }
        }
        *max_row_index = tmp_row;
        return;
    }
};
