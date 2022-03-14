/*
    This file aims to tracking the data flow of some root functions inside MM allocation functions,
        and record the propagated return value and parameters.
    The inter-procedural propagation of data flow are achieved by recursively call this plugins via other python scripts.
    The works of data flow merge, validation and MOS generation are implementated in python scripts.
*/
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
#include <deque>
#include <map>
#include <set>
using namespace clang;


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

std::map<std::string,std::set<std::string>> call_graph;
std::string input_path,output_path; 
std::set<std::string> visited;


bool read_visited(){
    std::ifstream in_file;

    in_file.open("/tmp/visited", std::ios::in);
    if(!in_file){llvm::errs()<<"open visited file error!\n";return false;}
    while(!in_file.eof())
    {
        std::string func_name;
        in_file >> func_name;
        visited.insert(func_name);
    }
    in_file.close();
    return true;
}


bool read_call_graph()
{
    std::ifstream in_file;

    in_file.open(input_path,std::ios::in);
    if(!in_file){llvm::errs()<<"open file error!\n";return false;}
    while(!in_file.eof())
    {
        std::string caller;
        in_file>>caller;
        if(caller.length() <=3) break;
        if(debug)llvm::errs()<<"caller:\t"<<caller<<"\n";
        std::set<std::string> callees;
        while(1)
        {
            std::string callee;
            in_file>>callee;
            if(callee == "-")
                break;
            callees.insert(callee);
            if(debug)llvm::errs()<<"callee:\t"<<callee<<"\n";
        }
        call_graph.insert(std::pair<std::string,std::set<std::string>>(caller,callees));
    }
    in_file.close();
    return true;
}

void trim(std::string &s) 
{
    if (s.empty()) 
    {
        return ;
    }
    s.erase(0,s.find_first_not_of(" "));
    s.erase(s.find_last_not_of(" ") + 1);
}


namespace{
    typedef FIFOWorkStack<const Stmt*> StmtWorkStack;

    
    std::set<std::string> getParamNames(FunctionDecl* FD){
            std::set<std::string> param_set;
            for (auto item: FD->parameters())
            {
                std::string current_type = QualType(item->getOriginalType().getTypePtr()->getUnqualifiedDesugaredType(), 0).getAsString();
                if (current_type.find("**") == std::string::npos) continue;

                std::string current_name = item->getNameAsString();
                param_set.insert(current_name);
            }
            return param_set;
        }



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

                StmtWorkStack temp_workstack;
                for (auto stmt : current_stmt->children())
                {
                    if(stmt)temp_workstack.push(stmt);
                }
                while(!temp_workstack.empty())
                {
                    auto temp_stmt = temp_workstack.pop();
                    workstack.push(temp_stmt);
                }
            }
            return "";
        }

        std::string GetBaseName(std::string member_name){
            std::string basename = "";
            for(int i=0;i<member_name.length();i++)
            {
                if (member_name[i] == '.' || member_name[i] == '-') break;
                basename += member_name[i];
            }
            return basename;
        }

        const MemberExpr* getMemberExpr(const Expr* expr){
            StmtWorkStack workstack;
            workstack.push(expr);
            while(!workstack.empty())
            {
                auto current_stmt = workstack.pop();
                if (auto memberExpr = dyn_cast<MemberExpr>(current_stmt))
                    {
                        return memberExpr;
                    }

                StmtWorkStack temp_workstack;
                for (auto stmt : current_stmt->children())
                {
                    if(stmt)temp_workstack.push(stmt);
                }
                while(!temp_workstack.empty())
                {
                    auto temp_stmt = temp_workstack.pop();
                    workstack.push(temp_stmt);
                }
            }
            return NULL;
        }

        std::string getRecurrentMemberName(const MemberExpr *memberExpr){
            StmtWorkStack workstack;
            workstack.push(memberExpr);
            std::string name = "";
            std::string base_name;
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
                if (auto declRefExpr = dyn_cast<DeclRefExpr>(current_stmt))
                    base_name = declRefExpr->getFoundDecl()->getNameAsString();

                for (auto stmt : current_stmt->children())
                    {if(stmt)workstack.push(stmt);break;}
            }
            //Expr *baseExpr = memberExpr->getBase();// has bug.
            //std::string base_name = GetExprValue(baseExpr);
            return base_name + name;
        }

    

        template<typename T>  std::string getFunctionName(const T* FD){
            if (auto fd = dyn_cast<FunctionDecl>(FD))
                return fd->getNameAsString();
            else
            if (auto fd = dyn_cast<VarDecl>(FD))
                return fd->getNameAsString();
            else
            if (auto fd = dyn_cast<ValueDecl>(FD))
                return fd->getNameAsString();
        }

         std::string FindCallExpr(const Stmt* expr){
            StmtWorkStack workstack;
            workstack.push(expr);

            while(!workstack.empty())
            {
                auto current_stmt = workstack.pop();
                if(auto callExpr = dyn_cast<CallExpr>(current_stmt))
                {
                    auto CD = callExpr->getCalleeDecl();
                    if (!CD) continue;;
                    return getFunctionName(CD);
                }

                StmtWorkStack temp_workstack;
                for (auto stmt : current_stmt->children())
                {
                    if(stmt)temp_workstack.push(stmt);
                }
                while(!temp_workstack.empty())
                {
                    auto temp_stmt = temp_workstack.pop();
                    workstack.push(temp_stmt);
                }
            }
            return "";
        }

        bool CheckVarDecl(const VarDecl *varDecl,std::string callee){
            if(!varDecl->hasInit()) return false;
            auto expr = varDecl->getInit();
            if(!expr) return false;


            std::string current_funcname = FindCallExpr(expr);
            if (current_funcname == callee)
                return true;
            else
                return false;
        }

        bool isAddressOperator(const Expr* expr)
        {
            StmtWorkStack workstack;
            workstack.push(expr);
            int deepth = 0;

            while(!workstack.empty())
            {
                auto current_stmt = workstack.pop();
                if(auto unaryOperator = dyn_cast<UnaryOperator>(current_stmt))
                {
                    
                    std::string name = unaryOperator->getOpcodeStr(unaryOperator->getOpcode()).str();
                    //llvm::errs()<<"unaryOperator:\t"<<name<<"\n";
                    if (name == "&")
                    return true;
                    else
                    return false;
                }
                for (auto stmt : current_stmt->children())
                    {
                        if(stmt)workstack.push(stmt); deepth++;break;
                    }
                
                if(deepth >3)break;
            }
            return false;
        }

        int CheckRetrenStmt(const ReturnStmt* returnStmt,std::string callee,std::set<std::string> &ret_value_set){
            auto expr = returnStmt->getRetValue();
            if(!expr) return -1;
            std::string current_funcname = FindCallExpr(expr);
            if (current_funcname == "")
            {   
                bool addres_operator_flag = isAddressOperator(expr);
                std::string value;
                if(auto memberExpr = getMemberExpr(expr))
                    value = getRecurrentMemberName(memberExpr);
                else
                    value = GetExprValue(expr);
                if (value == "")
                    return -1; // Found no callee and no return value.
                else
                    {
                        if (addres_operator_flag) value = "&" + value;
                        ret_value_set.insert(value);
                        return 1; // Found a return value;
                    }
            }
            else if (current_funcname == callee)
                    return 2; // Found a matched callee.
            else return -1;
        }

        bool  CheckBinaryOperator(const BinaryOperator* binaryOperator,std::set<std::string> &var_name_set,std::string callee){
            if (!binaryOperator->isAssignmentOp()) return false;
            Expr* left_expr = binaryOperator->getLHS();
            Expr* right_expr = binaryOperator->getRHS();

            std::string right_name;
            if(auto memberExpr = getMemberExpr(right_expr))
                right_name = getRecurrentMemberName(memberExpr);
            else
                right_name = GetExprValue(right_expr);
            
            std::string left_name;
            if(auto memberExpr = getMemberExpr(left_expr))
                left_name = getRecurrentMemberName(memberExpr);
            else
                left_name = GetExprValue(left_expr);

            std::string func_name = FindCallExpr(right_expr);
            if (func_name == callee)
            {
                var_name_set.insert(left_name);
                if(debug)llvm::errs()<<"Binary Operator, left name is :\t"<<left_name<<"\n";
                return true;
            }
            else if(func_name != "") return false;

            if (var_name_set.find(right_name) == var_name_set.end()) return false;
            std::string left_basename = GetBaseName(left_name);
            if (left_basename == right_name) return false;
            var_name_set.insert(left_name);
            if(debug)llvm::errs()<<"Binary Operator, left name is :\t"<<left_name<<"\n";
            if(debug)llvm::errs()<<"Binary Operator, Right name is :\t"<<right_name<<"\n";
            return true;
        }

        void debug_output(std::set<std::string> var_name_set,std::set<std::string> ret_value_set){
            std::set<std::string>::iterator iter;
            for(iter = var_name_set.begin();iter != var_name_set.end();iter++)
            {
                llvm::errs()<<"Pass Variable Name:\n"<<*iter<<"\n";
            }
            for(iter = ret_value_set.begin();iter != ret_value_set.end();iter++)
            {
                llvm::errs()<<"All Return Values:\n"<<*iter<<"\n";
            }
        }

        void save_result(std::set<std::string> var_name_set,std::set<std::string> ret_value_set,std::string funcname,int direct_return_flag,std::string callee){
            std::ostringstream raw_str;
            std::set<std::string>::iterator iter;
            raw_str<<"{\"caller\": \""<<funcname<<"\","<<"\"callee\": \""<<callee<<"\","
               <<"\"passed_params\": [\"";
            for(iter = var_name_set.begin();iter != var_name_set.end();iter++)
            {
                raw_str<<*iter<<"\",\"";
            }
            raw_str<<"<nops>\"], \"return_values\": [\"";
            for(iter = ret_value_set.begin();iter != ret_value_set.end();iter++)
            {
                raw_str<<*iter<<"\",\"";
            }
            raw_str<<"<nops>\"], \"direct_return\":"<<direct_return_flag<<"}\n";
            std::ofstream out;
            out.open(output_path,std::ios::app);
            out<<raw_str.str();
            out.close();
            if(debug){llvm::errs()<<"\n Final output:\n"<<raw_str.str()<<"\n";}

            visited.insert(funcname);
            //out.open("/tmp/visited", std::ios::app);
            //out<<funcname + "\n";
            //out.close();
        }

    void ProcessACallee(Stmt* funcBody,std::string current_name,std::string callee_name)
    {
        std::set<std::string> var_name_set;
        std::set<std::string> ret_value_set;
        int direct_return_flag = 0;

        // Iterate with all AST nodes.
        if(debug){llvm::errs()<<"callee name:\t"<<callee_name<<"\n";}
        StmtWorkStack workstack;
        workstack.push(funcBody);
        while(!workstack.empty()){
            auto current_stmt = workstack.pop();
            if(auto declStmt = dyn_cast<DeclStmt>(current_stmt)) // char *buf = malloc(20);
            {
                if (!declStmt->isSingleDecl())continue;
                auto varDecl = dyn_cast<VarDecl>(declStmt->getSingleDecl());
                if (!varDecl) continue;

                if(CheckVarDecl(varDecl,callee_name)){
                    std::string varName = varDecl->getNameAsString();
                    var_name_set.insert(varName);
                    if(debug){llvm::errs()<<"Found a VarDecl, var name is:\t"<<varName<<"\n";}
                }
            }
            if (auto binaryOperator = dyn_cast<BinaryOperator>(current_stmt))
            {
                CheckBinaryOperator(binaryOperator,var_name_set,callee_name);
            }

            if(auto returnStmt = dyn_cast<ReturnStmt>(current_stmt))
            {
                int flag = CheckRetrenStmt(returnStmt,callee_name,ret_value_set);
                if (flag == 2)
                    {
                        //llvm::errs()<<"Find a directly returned callee\n";
                        direct_return_flag = 1;
                    }
            }
            StmtWorkStack temp_workstack;
            for (auto stmt : current_stmt->children())
            {
                if(stmt)temp_workstack.push(stmt);
            }
            while(!temp_workstack.empty())
            {
                auto temp_stmt = temp_workstack.pop();
                workstack.push(temp_stmt);
            }
        }
        save_result(var_name_set,ret_value_set,current_name,direct_return_flag,callee_name);
    }
    
    class FindFunctionVisitor : public clang::RecursiveASTVisitor<FindFunctionVisitor>{
        //ASTContext* Context;
        public:
        //FindFunctionVisitor(ASTContext* Context): Context(Context){}

        
        bool VisitFunctionDecl(FunctionDecl* FD){
            std::string current_funcname;
            if (FD) {
                FD = FD->getDefinition() == nullptr ? FD : FD->getDefinition();
                if (!FD->isThisDeclarationADefinition()) return true;
                if (!FD->doesThisDeclarationHaveABody()) return true;
                current_funcname = FD->getQualifiedNameAsString();

                std::set<std::string>::iterator iter0 = visited.find(current_funcname);
                if (iter0 != visited.end()) return true;
            
                // Check wheter current function in call graph.
                std::map<std::string,std::set<std::string>>::iterator iter = call_graph.find(current_funcname);
                if(iter == call_graph.end())
                    return true;
                
                auto funcBody = FD->getBody();
                if(!funcBody)return true;
                if(debug){llvm::errs()<<"Current funcname:\t"<<current_funcname<<"\n";}

                std::set<std::string> callees = iter->second;
                std::set<std::string>::iterator callee_iter;

                for(callee_iter = callees.begin();callee_iter != callees.end(); callee_iter++)
                {
                    ProcessACallee(funcBody,current_funcname,*callee_iter);
                }
                return true;
            }
        }

    };



    class MemoryDataFlowConsumer : public ASTConsumer {
        FindFunctionVisitor Visitor;
    public:
        explicit MemoryDataFlowConsumer(ASTContext* Context) {}

        virtual void HandleTranslationUnit(clang::ASTContext &Context) {
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
            
            if(args.size() <2)
            {
                llvm::errs()<< "Lack input and output path!\n";
                return false;
            }
            input_path = args[0];
            output_path = args[1];
            bool ret = read_call_graph();
            if(!ret) return false;
            //ret = read_visited();
            //if(!ret) return false;
            return true;
        }
    };
}


static FrontendPluginRegistry::Add<MemoryDataFlowAction>
        X("point-memory", "Point the dataflow of memory functions.");
