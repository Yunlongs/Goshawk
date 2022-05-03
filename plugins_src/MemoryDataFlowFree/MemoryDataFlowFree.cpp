#include <utility>
#include<sstream>
#include<iostream>
#include<fstream>
#include "clang/Frontend/FrontendPluginRegistry.h"
#include "clang/AST/AST.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Sema/Sema.h"
#include "llvm/Support/raw_ostream.h"
#include <assert.h>
#include <cstdlib>
#include <vector>
#include <stack>
#include <map>
#include <set>
#include "json.hpp"

using json = nlohmann::json;
using namespace clang;

//command: new_cmd = "clang -fsyntax-only -Xclang -load -Xclang ./MemoryDataFlowFree.so -Xclang -plugin -Xclang point-memory-free  -Xclang -plugin-arg-point-memory-free -Xclang 1 -Xclang -plugin-arg-point-memory-free -Xclang ./candidate_free.txt -Xclang -plugin-arg-point-memory-free -Xclang ./seed_free.txt -Xclang -plugin-arg-point-memory-free -Xclang ./memory_flow_free.json -Xclang -plugin-arg-point-memory-free -Xclang ./last_step_mos.json -Xclang -plugin-arg-point-memory-free -Xclang ./visited.txt"
//

/*Set `true` to enter debug mode*/
bool debug = false;


template<class Data>
class FIFOWorkStack {
    typedef std::set<Data> DataSet;
    typedef std::stack<Data> DataStack;
public:
    FIFOWorkStack() {}

    ~FIFOWorkStack() {}

    inline bool empty() const {
        return data_stack.empty();
    }

    inline bool find(Data data) const {
        return (data_set.find(data) == data_set.end() ? false : true);
    }

    inline bool push(Data data) {
        if (data_set.find(data) == data_set.end()) {
            data_stack.push(data);
            data_set.insert(data);
            return true;
        }
        else
            return false;
    }

    inline Data pop() {
        assert(!empty() && "work list is empty");
        Data data = data_stack.top();
        data_stack.pop();
        data_set.erase(data);
        return data;
    }

    inline void clear() {
        data_stack.clear();
        data_set.clear();
    }

private:
    DataSet data_set;	///< store all data in the work list.
    DataStack data_stack;	///< work stack using std::stack.
};


std::string candidate_free_path, seed_free_path, mos_seed_path, mos_free_outpath, step, visited_file_path;
std::set<std::string> visited, candidate_free_set;
std::map<std::string,int> seed_free_map;
std::map<std::string,std::string> mos_funcs;
typedef std::vector<const Expr*> ExprVector;

/*
    Maintain a visited function name set. Every time we process a file,
    we read this visited file first. If we encounter a function which
    already in this set, there is no need to process this function again.    
*/
bool read_visited(){
    std::ifstream in_file;

    in_file.open(visited_file_path, std::ios::in);
    if(!in_file){llvm::errs()<<"open visited file error! Check this path:"<<visited_file_path<<"\n";return false;}
    std::string readline;
    while(getline(in_file,readline))
    {
        visited.insert(readline);
    }
    in_file.close();
    return true;
}

/*
    Here we read two sets: `sedd_free_set` and `free_set`.
    seed_free_set: initial seed function, such as free,kfree.
    free_set: high semantics similarity with funciton prototype. such as dmabuf_gem_object_free.
*/
bool read_free_function()
{
    std::ifstream in_file;

    in_file.open(seed_free_path,std::ios::in);
    if(!in_file){llvm::errs()<<"open seed file error! Check this path:"<<seed_free_path<<"\n";return false;}
    while(!in_file.eof())
    {
        std::string func_name;
        in_file>>func_name;
        int index;
        in_file>>index;
        seed_free_map.insert(std::pair<std::string,int>(func_name,index));
    }
    in_file.close();

    in_file.open(candidate_free_path, std::ios::in);
    if(!in_file){llvm::errs()<<"open candidate free file error! Check this path:"<<candidate_free_path<<"\n";return false;}
    while(!in_file.eof())
    {
        std::string func_name;
        in_file>>func_name;
        candidate_free_set.insert(func_name);
    }
    in_file.close();
    return true;
}


bool read_mos_free()
{
    std::ifstream in_file;

    in_file.open(mos_seed_path,std::ios::in);
    if(!in_file){llvm::errs()<<"open mos free file error! Check this path:"<<mos_seed_path<<"\n";return false;}
    
    std::string readline;
    while(getline(in_file,readline))
    {
        if(readline.length() <=3) break;
        auto func = json::parse(readline);
        std::string funcname = func["funcname"];
        if(debug)llvm::errs()<<"caller:\t"<<funcname<<"\n";
        mos_funcs.insert(std::pair<std::string,std::string>(funcname,readline));
    }
    in_file.close();
    return true;
}

namespace{
    typedef FIFOWorkStack<const Stmt*> StmtWorkStack;

    std::vector<std::string> getParamNames(FunctionDecl* FD){
            std::vector<std::string> param_set;
            for (auto item: FD->parameters())
            {
                //std::string current_type = QualType(item->getOriginalType().getTypePtr()->getUnqualifiedDesugaredType(), 0).getAsString();
                //if (current_type.find("*") == std::string::npos) continue;

                std::string current_name = item->getNameAsString();
                param_set.push_back(current_name);
            }
            return param_set;
        }

        /*
            Get the DeclRefExpr's Name.
        */
        std::string GetExprValue(const Stmt *expr){
            StmtWorkStack workstack;
            workstack.push(expr);
            while(!workstack.empty())
            {
                auto current_stmt = workstack.pop();
                if(auto declRefExpr = dyn_cast<DeclRefExpr>(current_stmt))
                {
                    
                    return declRefExpr->getFoundDecl()->getNameAsString();
                }
                for (auto stmt : current_stmt->children())
                        if(stmt)workstack.push(stmt);
            }
            return "";
        }

    
        /*
            Get the CallExpr's name which includes indirect callee.
        */
        template<typename T>  std::string getFunctionName(const T* FD){
            if (auto fd = dyn_cast<FunctionDecl>(FD))
                return fd->getNameAsString();
            /*
            else
            if (auto fd = dyn_cast<VarDecl>(FD))
                return fd->getNameAsString();
            else
            if (auto fd = dyn_cast<ValueDecl>(FD))
                return fd->getNameAsString();
            */
            return "";
        }


        bool isCallExpr(const Expr* expr){
            expr = expr->IgnoreCasts();
            if(dyn_cast<CallExpr>(expr)) return true;
            return false;
        }

        /*
            If this Expr is a structure member or array, then we get
            the corresponding expr.
        */
        const Expr* getMemberOrArrayExpr(const Expr* expr){
            StmtWorkStack workstack;
            workstack.push(expr);
            while(!workstack.empty())
            {
                auto current_stmt = workstack.pop();
                if (auto memberExpr = dyn_cast<MemberExpr>(current_stmt))
                    {
                        return memberExpr;
                    }
                if (auto arraySubscriptExpr = dyn_cast<ArraySubscriptExpr>(current_stmt))
                {
                    return arraySubscriptExpr;
                }
                for (auto stmt : current_stmt->children())
                    if(stmt)workstack.push(stmt);
            }
            return NULL;
        }

        /* If this expr contains no Memberexpr, return a empty vector.
            Otherwise, push the top-level Member Expr to this vector.
        */
        ExprVector  getMemberOrArrayExprVector(const Expr* expr){
            ExprVector expr_vector;
            StmtWorkStack workstack;
            workstack.push(expr);
            while(!workstack.empty())
            {
                auto current_stmt = workstack.pop();
                if (auto memberExpr = dyn_cast<MemberExpr>(current_stmt))
                    {
                        expr_vector.push_back(memberExpr);
                        continue;
                    }
                if (auto arraySubscriptExpr = dyn_cast<ArraySubscriptExpr>(current_stmt))
                {
                    expr_vector.push_back(arraySubscriptExpr);
                    continue;
                }
                for (auto stmt : current_stmt->children())
                    if(stmt)workstack.push(stmt);
            }
            return expr_vector;
        }

        /*
            According to the top-lvel memberExpr Or arraySubscriptExpr, 
            we retreive it's  full name. such as :block->buf.ptr[0].
        */
        std::string getRecurrentFullName(const Expr *memberOrArrayExpr){
            StmtWorkStack workstack;
            workstack.push(memberOrArrayExpr);
            std::string name = "";
            while(!workstack.empty())
            {
                auto current_stmt = workstack.pop();
                if (auto memberExpr = dyn_cast<MemberExpr>(current_stmt))
                {
                    std::string current_name = memberExpr->getFoundDecl()->getNameAsString();
                    if (memberExpr->isArrow())
                    {
                        name = "->" + current_name + name;
                    }
                    else{
                        name = "." + current_name + name;
                    }
                }
                if (dyn_cast<ArraySubscriptExpr>(current_stmt))
                {
                    name = "[i]" + name;
                }
                for (auto stmt : current_stmt->children())
                    if(stmt)workstack.push(stmt);
            }
            const Expr *baseExpr;
            if (auto memberExpr = dyn_cast<MemberExpr>(memberOrArrayExpr))
                baseExpr = memberExpr->getBase();
            else if (auto arraySubscriptExpr = dyn_cast<ArraySubscriptExpr>(memberOrArrayExpr))
                baseExpr = arraySubscriptExpr->getBase();
            else 
                return "";
            std::string base_name = GetExprValue(baseExpr);
            return base_name + name;
        }

        /*
        1. Check the right expression of BinaryOperator and VarDecl.
        2. Record the member name or variable names appears in this expr.
        */
        std::string CheckRightExpr(const Expr *expr){
            if (isCallExpr(expr)) return "";
            StmtWorkStack workstack;
            workstack.push(expr);
            ExprVector member_vector = getMemberOrArrayExprVector(expr);
            if (!member_vector.empty())
            {
                std::string member_name;
                for(size_t i=0;i<member_vector.size();i++)
                {
                    auto memberExpr = member_vector[i];
                    member_name = getRecurrentFullName(memberExpr);
                    //member_name_set.insert(member_name);
                    //if (debug){llvm::errs()<<"Right Expr member name: "<<member_name<<"\n";}
                }
                return member_name;
  
            }
            while(!workstack.empty())
            {
                auto current_stmt = workstack.pop();
                
                if(auto declRefExpr = dyn_cast<DeclRefExpr>(current_stmt))
                {
                    std::string var_name =  declRefExpr->getFoundDecl()->getNameAsString();
                    return var_name;
                    //var_name_set.insert(var_name);
                    //if (debug){llvm::errs()<<"Right Expr var name: "<<var_name<<"\n";}
                }
                for (auto stmt : current_stmt->children())
                        if(stmt)workstack.push(stmt);
            }
            return "";
        }

        std::string GetBaseName(std::string member_name){
            std::string basename = "";
            for(size_t i=0;i<member_name.length();i++)
            {
                if (member_name[i] == '.' || member_name[i] == '-' || member_name[i] == '[') break;
                basename += member_name[i];
            }
            return basename;
        }

        std::string ChangeBaseName(std::string member_name,std::string new_basename)
        {
            std::string old_member_name;
            size_t i;
            for(i=0;i < member_name.length();i++)
            {  
                if (member_name[i] == '-' || member_name[i] == '.') break;
            }
            for(;i<member_name.length();i++)
                old_member_name += member_name[i];
            return new_basename + old_member_name;
        }


        /*
            For some cases in Linux Kernel, the macro `container_of` is used to 
            restoring parert structure by computing address offset. These could 
            casue false positives.
        */
        bool IsContainer(const Expr* expr)
        {
            StmtWorkStack workstack;
            workstack.push(expr);
            while(!workstack.empty())
            {
                auto current_stmt = workstack.pop();
                if (auto declRef = dyn_cast<DeclRefExpr>(current_stmt))
                    {
                        std::string name = GetExprValue(declRef);
                        if (name == "__mptr")
                            return true;
                    }
                for (auto stmt : current_stmt->children())
                    if(stmt)workstack.push(stmt);
            }

            return false;
        }

        bool isMemberName(std::string name)
        {
            for (size_t i = 0; i < name.length(); i++)
            {
                if (name[i] == '.' || name[i] == '-')
                    return true;
            }

            return false;
        }


        bool ModelMosFunction(const CallExpr* callExpr, std::set<std::string> &var_name_set, std::set<std::string> &member_name_set, std::string mos)
        {
            auto callee_func = json::parse(mos);

            if (debug){llvm::errs()<<"parsing the json string.\n"<<callee_func.dump()<<"\n";}

            auto param_name = callee_func["param_names"];
            if (debug) {llvm::errs()<<"param_name dump:\t";param_name.dump();llvm::errs()<<"\n";}
            if (param_name.size())
            {
                //for(std::string curr_name :param_name)
                for (size_t i =0 ;i<param_name.size();i++)
                {
                    //int arg_index = GetArgIndex(CD, curr_name);
                    int arg_index = param_name[i];
                    std::string curr_name = param_name[++i];
                    if (debug) {llvm::errs()<<"Param Name:\t"<<curr_name<<"\t\tindex:"<<arg_index<<"\n";}
                    if (arg_index == -1) return false;
                    const Expr* arg = callExpr->getArg(arg_index);
                    std::string name = CheckRightExpr(arg);
                    if (name == "") return false;
                    if (isMemberName(name))
                    {
                        member_name_set.insert(name);
                        if (debug){llvm::errs()<<"Insert a new member:\t"<<name<<"\n";}
                    }
                        
                    else
                    {
                        var_name_set.insert(name);
                        if (debug){llvm::errs()<<"Insert a var member:\t"<<name<<"\n";}
                    }
                }
            }

            auto member_name_list = callee_func["member_name"];
            for (size_t i =0; i<member_name_list.size(); i++)
            {
                int arg_index = member_name_list[i];
                std::string member_name = member_name_list[++i];
                //std::string base_name = GetBaseName(member_name);
                //int arg_index = GetArgIndex(CD, base_name);
                if (arg_index == -1) return false;
                const Expr *arg = callExpr->getArg(arg_index);
                std::string arg_name;
                if (const Expr *member_expr  = getMemberOrArrayExpr(arg))
                {
                    arg_name = getRecurrentFullName(member_expr);
                }
                else
                {
                    arg_name = GetExprValue(arg);
                }
                std::string new_member_name = ChangeBaseName(member_name,arg_name);
                member_name_set.insert(new_member_name);
                if (debug) {llvm::errs()<<"Insert a new member name:\t"<<new_member_name<<"\n";}
            }
            return true;
        }



        // 1. Get the Callee's function name and check whether this callee is in seed free function set.
        // 
        bool CheckCallExpr(const CallExpr* callExpr, std::set<std::string> &var_name_set, std::set<std::string> &member_name_set, std::string caller_name){
            auto CD = callExpr->getCalleeDecl();
            if (!CD) return false;
            std::string func_name = getFunctionName(CD);
            if (func_name == caller_name) return false; // avoid the recursion call
            if (seed_free_map.find(func_name) != seed_free_map.end())
            {
                if (debug){llvm::errs()<<"call a free function:\t"<<func_name<<"\n";}
                int arg_size = callExpr->getNumArgs();
                int arg_index = seed_free_map.find(func_name)->second;
                if (arg_index >= arg_size) return false;

                const Expr* arg = callExpr->getArg(arg_index);
                std::string name = CheckRightExpr(arg);
                if (name == "") return false;
                if (isMemberName(name))
                {
                    member_name_set.insert(name);
                    if (debug){llvm::errs()<<"Insert a new member:\t"<<name<<"\n";}
                }
                else
                {
                    var_name_set.insert(name);
                    if (debug){llvm::errs()<<"Insert a var member:\t"<<name<<"\n";}
                }
                return true;
            }
            else if (mos_funcs.find(func_name) != mos_funcs.end())
            {
                if (debug) {llvm::errs()<<"call a mos function:\t"<<func_name<<"\n";}
                return ModelMosFunction(callExpr, var_name_set, member_name_set, mos_funcs.find(func_name)->second);
            }

            return false;
        }

        /**/
        bool CheckBinaryOperator(const BinaryOperator* binaryOperator,std::set<std::string> &var_name_set,std::set<std::string> &member_name_set,std::map<std::string,std::string> &name_map){
            if (!binaryOperator->isAssignmentOp()) return false;
            Expr* left_expr = binaryOperator->getLHS();
            Expr* right_expr = binaryOperator->getRHS();
            if(IsContainer(right_expr)) return false;

            if (auto member_expr = getMemberOrArrayExpr(left_expr))
            {
                std::string member_name = getRecurrentFullName(member_expr);
                if (member_name_set.find(member_name) == member_name_set.end()) return false;
                std::string new_name = CheckRightExpr(right_expr);
                if(new_name != "")
                {
                    if(isMemberName(new_name))
                    {
                        member_name_set.insert(new_name);
                        if (debug){llvm::errs()<<"Insert a new member:\t"<<new_name<<"\n";}
                    }
                        
                    else
                    {
                        var_name_set.insert(new_name);
                        if (debug){llvm::errs()<<"Insert a new var name:\t"<<new_name<<"\n";}
                    }
                    
                    name_map.insert(std::pair<std::string,std::string>(member_name,new_name));
                    return true;
                }
            }

            /*
                ptrb = ptra;
                free(ptrb->field);
            */
            std::string left_value = GetExprValue(left_expr);
            //if (var_name_set.find(left_value) == var_name_set.end()) return false;
            int flag = 0; // 0: this var in var_name_set.  1: this var in member_name_set.
            if (var_name_set.find(left_value) == var_name_set.end())
            {
                 // if the variable in the freed variable set. 
                for (std::string member_name : member_name_set)
                    if (GetBaseName(member_name) == left_value)
                        {flag = 1 ;break;}
                if(!flag)
                    return false;
            }

            std::string new_name = CheckRightExpr(right_expr);
            if(debug){llvm::errs()<<"Binary Operator: "<<left_value<<" = "<<new_name<<"\n";}
            if(new_name == "") return true;
            
            if(flag == 0) //this var in var_name_set :struct kyber_hctx_data *khd = hctx->sched_data; free(khd);
            {
                if(isMemberName(new_name))
                {
                    member_name_set.insert(new_name);
                    if (debug){llvm::errs()<<"Insert a new member:\t"<<new_name<<"\n";}
                }
                    
                else
                {
                    var_name_set.insert(new_name);
                    if (debug){llvm::errs()<<"Insert a new var name:\t"<<new_name<<"\n";}
                }    
            }

            name_map.insert(std::pair<std::string,std::string>(left_value,new_name));
            return true;
        }

        /*
            When we meet a variable declaration, we need to check this defined
            variable whether is the variables we later freed.
            If it is, then we record the member or variable that 
            passed in this declaration.
        */
        bool CheckVarDecl(const VarDecl *varDecl,std::set<std::string> &var_name_set,std::set<std::string> &member_name_set,std::map<std::string,std::string> &name_map){
            if(!varDecl->hasInit()) return false;
            auto expr = varDecl->getInit();
            if (!expr) return false;
            if (IsContainer(expr)) return false;

            std::string varName = varDecl->getNameAsString();
            int flag = 0; // 0: this var in var_name_set.  1: this var in member_name_set.
            if (var_name_set.find(varName) == var_name_set.end())
            {
                 // if the variable in the freed variable set.
                for (std::string member_name : member_name_set)
                    if (GetBaseName(member_name) == varName)
                        {flag = 1 ;break;}
                if(!flag)
                    return false;
            }

            std::string new_name = CheckRightExpr(expr);
            if(debug){llvm::errs()<<"Find a target Declaration, the corresponded variable name is :"<<varName<<"\n";}
            if(new_name == "") return true;
            if(flag == 0) //this var in var_name_set :struct kyber_hctx_data *khd = hctx->sched_data; free(khd);
            {
                if(isMemberName(new_name))
                {
                    member_name_set.insert(new_name);
                    if (debug){llvm::errs()<<"Insert a new member:\t"<<new_name<<"\n";}
                }
                    
                else
                {
                    var_name_set.insert(new_name);
                    if (debug){llvm::errs()<<"Insert a new var name:\t"<<new_name<<"\n";}
                }
                
            }
            // if flag ==1, we only need record the name binding. : struct kyber_hctx_data *khd = hctx->sched_data; free(khd->kcq);
            name_map.insert(std::pair<std::string,std::string>(varName,new_name));
            return true;
        }


        /*
            What does this function do? 
        */
        std::vector<int> CheckConsistent(std::vector<std::string> param_set,std::set<std::string> &var_name_set,
                            std::set<std::string> &member_name_set,std::map<std::string,std::string> &name_map){
            std::set<std::string>::iterator iter;

            for(std::set<std::string>::iterator iter = member_name_set.begin();iter!=member_name_set.end();iter++)
            {
                std::string basename = GetBaseName(*iter);
                if(debug){llvm::errs()<<"member: "<<*iter<<"\t,base name:\t"<<basename<<"\n";}
                std::map<std::string,std::string>::iterator iter_map = name_map.find(basename);
                if (iter_map != name_map.end())
                {
                    if(iter_map->first == GetBaseName(iter_map->second)) continue;
                    std::string new_name = ChangeBaseName(*iter,iter_map->second);
                    member_name_set.insert(new_name);
                }
            }

            int index = 0,flag = 0;
            std::vector<int> param_index;
            std::vector<std::string>::iterator v_iter;
            for (v_iter = param_set.begin();v_iter != param_set.end();v_iter++,index++)
            {
                if (var_name_set.find(*v_iter) != var_name_set.end())
                {
                    param_index.push_back(index);
                    flag = 1;
                }
            }
            
            index = 0;
            for (iter = member_name_set.begin();iter != member_name_set.end();iter++)
            {
                std::string basename = GetBaseName(*iter);
                //if (param_set.find(basename) != param_set.end())
                for (auto name : param_set)
                    if (name == basename)
                        flag =1;
            }
            param_index.push_back(flag);
            return param_index;
        }

        /*
            To emit the detailed set information.
        */
        void debug_output(std::set<std::string> var_name_set,std::set<std::string> param_set,std::string funcname,int index,std::set<std::string> member_name_set,std::map<std::string,std::string> name_map){
            std::set<std::string>::iterator iter;
            llvm::errs()<<"\nFinal var_name_set:\n";
            for(iter = var_name_set.begin();iter != var_name_set.end();iter++)
            {
                llvm::errs()<<*iter<<"\t";
            }
            llvm::errs()<<"\n\nFinal member_name_set:\n";
            for(iter = member_name_set.begin();iter != member_name_set.end();iter++)
            {
                llvm::errs()<<*iter<<"\t";
            }
            llvm::errs()<<"\n\nFinal name mapping:\n";
            std::map<std::string,std::string>::iterator iter_map;
            for(iter_map = name_map.begin();iter_map!=name_map.end();iter_map++)
            {
                llvm::errs()<<iter_map->first<<":\t"<<iter_map->second<<"\n";
            }
        }

        void save_result(std::set<std::string> var_name_set,std::vector<std::string> param_set,std::string funcname,
        std::vector<int> param_index,std::set<std::string> member_name_set,std::map<std::string,std::string> name_map){
            std::ofstream out;
            visited.insert(funcname);
            out.open(visited_file_path, std::ios::app);
            out<<funcname<<std::endl;
            out.close();
            int index = param_index[param_index.size()-1];
            param_index.pop_back();
            if(!index) return;

            int i = 0;
            std::vector<std::pair<int,std::string>> param_name;

            for (std::vector<std::string>::iterator iter = param_set.begin();iter !=param_set.end();iter++,i++)
            {
                for (int curr_index : param_index)
                    if (i == curr_index)
                    {
                        std::string name = *iter;
                        param_name.push_back(std::pair<int, std::string>(curr_index,name));
                    }         
            }
            
            //if(debug){debug_output(var_name_set,param_set,funcname,index,member_name_set,name_map);}

            std::ostringstream raw_str;
            std::set<std::string>::iterator iter;
            raw_str<<"{\"funcname\": \""<<funcname<<"\",\"param_names\":[";
            for (auto pair : param_name)
            {
                int index = pair.first;
                std::string curr_name = pair.second;
                raw_str<<index<<",";
                raw_str<<"\""<<curr_name<<"\",";
            }
            raw_str<<"\"<nops>\"],";
            
            raw_str<<"\"member_name\": [";
            for(std::set<std::string>::iterator iter = member_name_set.begin(); iter != member_name_set.end(); iter++)
            {
                std::string basename = GetBaseName(*iter);
                for (size_t i=0;i<param_set.size();i++)
                {
                    if (param_set[i] == basename)
                    {
                        raw_str<<i<<",";
                        raw_str<< "\""<<*iter<<"\",";
                    }
                }
            }
            raw_str<<"\"<nops>\"]}\n";
            out.open(mos_free_outpath,std::ios::app);
            out<<raw_str.str();
            out.close();

        } 

    
    bool GetMemoryFlow(FunctionDecl* FD)
    {
        std::string current_funcname;
        FD = FD->getDefinition() == nullptr ? FD : FD->getDefinition();
        if (!FD->isThisDeclarationADefinition()) return true;
        if (!FD->doesThisDeclarationHaveABody()) return true;
        //llvm::errs()<<"initial handle_1\n";
        current_funcname = FD->getQualifiedNameAsString();

        // If this function is visited, then return.
        if (visited.find(current_funcname) != visited.end()) return true;
    
        // If this function not in candidate free set, then return.
        std::set<std::string>::iterator iter = candidate_free_set.find(current_funcname);
        if(iter == candidate_free_set.end())
            return true;
        
        if(debug){llvm::errs()<<"\n\ncurrent funcanme:\t"<<current_funcname<<"\n";}
        auto funcBody = FD->getBody();
        if(!funcBody)return true;

        std::set<std::string> var_name_set;
        std::set<std::string> member_name_set;
        std::map<std::string, std::string> name_map;
        

        // Iterate with all AST nodes.
        StmtWorkStack workstack;
        workstack.push(funcBody);
        while(!workstack.empty()){
            auto current_stmt = workstack.pop();
            if (auto callExpr = dyn_cast<CallExpr>(current_stmt))
            {           
                CheckCallExpr(callExpr,var_name_set,member_name_set, current_funcname);
            }
            if (auto binaryOperator = dyn_cast<BinaryOperator>(current_stmt))
            {
                CheckBinaryOperator(binaryOperator,var_name_set,member_name_set,name_map);
            }
            if (auto declStmt = dyn_cast<DeclStmt>(current_stmt))
            {
                if (!declStmt->isSingleDecl())continue;
                auto varDecl = dyn_cast<VarDecl>(declStmt->getSingleDecl());
                if (!varDecl) continue;
                CheckVarDecl(varDecl,var_name_set,member_name_set,name_map);
            }

            for (auto stmt : current_stmt->children())
                    if(stmt)workstack.push(stmt);
        }

        std::vector<std::string> param_set = getParamNames(FD);
        std::vector<int> param_index = CheckConsistent(param_set,var_name_set,member_name_set,name_map);
        save_result(var_name_set,param_set,current_funcname,param_index,member_name_set,name_map);
        return true;
    }


    class FindFunctionVisitor : public clang::RecursiveASTVisitor<FindFunctionVisitor>{
        //ASTContext* Context;
        public:
        //FindFunctionVisitor(ASTContext* Context): Context(Context){}

        bool VisitFunctionDecl(FunctionDecl* FD){
            if (FD) {
                return GetMemoryFlow(FD);
            }
            return true;
        }
    };



    class MemoryDataFlowConsumer : public ASTConsumer {
        FindFunctionVisitor Visitor;
    public:
        explicit MemoryDataFlowConsumer(ASTContext* Context) {}

        virtual void HandleTranslationUnit(clang::ASTContext &Context) override{
            //llvm::errs()<<"HandleTranslationUnit!\n";
            Visitor.TraverseDecl(Context.getTranslationUnitDecl());
        }
    };


    class MemoryDataFlowAction : public PluginASTAction {
    protected:
        std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                       llvm::StringRef) override {
            return std::make_unique<MemoryDataFlowConsumer>(&CI.getASTContext());
        }

        bool ParseArgs(const CompilerInstance &CI,
                       const std::vector<std::string> &args) override {
        /*
            Arguments List: step, candidate_free_path, seed_free_path, mos_seed_path, mos_free_outpath, visited_file_path.
        */
            
            if(args.size() <6)
            {
                llvm::errs()<< "Insufficient Arguments!\n Arguments List: step, candidate_free_path, seed_free_path, mos_seed_path, mos_free_outpath, visited_file_path.\n";
                return false;
            }
            step = args[0];
            candidate_free_path = args[1];
            seed_free_path = args[2];
            mos_seed_path = args[3];
            mos_free_outpath = args[4];
            visited_file_path = args[5];

            if (!read_visited())
                return false;

            if(!read_free_function()) // step1: According the seed functions, to generate the first mos functions.
                return false;
            
            if (step == "2") // According seed functions and already generated mos fucntions, to generate more MOS.
               if(!read_mos_free())
                    return false;
            return true;
        }
    };
}


static FrontendPluginRegistry::Add<MemoryDataFlowAction>
        X("point-memory-free", "Point the dataflow of memory functions about deallocation.");
