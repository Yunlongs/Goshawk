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

std::string input_path, output_path, visited_path; 
std::set<std::string> visited;
std::set<std::string> FreeFuncs;
std::vector<std::string> Result;
typedef std::vector<const Expr*> ExprVector;


bool read_visited(){
    std::ifstream in_file;

    in_file.open(visited_path, std::ios::in);
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

bool ReadFreeFuncs(){
    std::ifstream in_file;

    in_file.open(input_path, std::ios::in);
    if(!in_file){llvm::errs()<<"open free funcs file error!\n";return false;}
    while(!in_file.eof())
    {
        std::string func_name;
        in_file >> func_name;
        if (debug){llvm::errs()<<func_name<<"\n";}
        FreeFuncs.insert(func_name);
    }
    in_file.close();
    return true;
}




namespace{
    typedef FIFOWorkStack<const Stmt*> StmtWorkStack;

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

            bool isCallExpr(const Expr* expr){
            expr = expr->IgnoreCasts();
            if(auto callExpr = dyn_cast<CallExpr>(expr)) return true;
            return false;
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
                for (auto stmt : current_stmt->children())
                        if(stmt)workstack.push(stmt);
            }
            return "";
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
                if (auto arraySubscriptExpr = dyn_cast<ArraySubscriptExpr>(current_stmt))
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
    
    std::string CheckRightExpr(const Expr *expr){
            if (isCallExpr(expr)) return "";
            StmtWorkStack workstack;
            workstack.push(expr);
            ExprVector member_vector = getMemberOrArrayExprVector(expr);
            if (!member_vector.empty())
            {
                std::string member_name;
                for(int i=0;i<member_vector.size();i++)
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

    bool isMemberName(std::string name)
        {
            for (int i = 0; i < name.length(); i++)
            {
                if (name[i] == '.' || name[i] == '-')
                    return true;
            }

            return false;
        }

    bool CheckCallExpr(const CallExpr* callExpr, std::set<std::pair<int,std::string>> &var_name_set, std::map<std::string, std::string> &param_func_map){
            auto CD = callExpr->getCalleeDecl();
            if (!CD) return false;
            std::string func_name = getFunctionName(CD);
            if (FreeFuncs.find(func_name) == FreeFuncs.end()) return false;
            if (debug){llvm::errs()<<"free function:\t"<<func_name<<"\n";}
            int arg_size = callExpr->getNumArgs();
            for (int index=0; index<arg_size; index++)
            {
                const Expr* arg = callExpr->getArg(index);
                if(!arg->getType()->isPointerType())
                    continue;
                std::string name = CheckRightExpr(arg);
                if (name == "") return false;

                var_name_set.insert(std::pair<int, std::string>(index, name));
                param_func_map.insert(std::pair<std::string, std::string>(name,func_name));
                if (debug){llvm::errs()<<"Insert a var member:\t"<<name<<"\n";}
                
            }
            return true;
        }


        bool CheckBinaryOperator(const BinaryOperator* binaryOperator,std::set<std::pair<int, std::string>> &var_name_set, std::map<std::string, std::string> &param_func_map){
            if (!binaryOperator->isAssignmentOp()) return false;
            Expr* left_expr = binaryOperator->getLHS();
            std::string current_left_name = "";

            if (auto member_expr = getMemberOrArrayExpr(left_expr))
            {
                std::string member_name = getRecurrentFullName(member_expr);
                current_left_name = member_name;
            }
            else{
                current_left_name = GetExprValue(left_expr);
            }
            
            if (current_left_name == "")
                return false;
            if(debug) {llvm::errs()<<"binary operator: "<<"\t"<<current_left_name<<"\n";}
            
            for (auto pair: var_name_set)
            {
                int index = pair.first;
                std::string name = pair.second;
                if (name == current_left_name)
                {
                    std::ostringstream raw_str;
                    auto iter = param_func_map.find(name);
                    std::string current_callee = iter->second;
                    raw_str << current_callee<<"\t"<<index<<"\n";
                    std::string result = raw_str.str();
                    raw_str.clear();
                    Result.push_back(result);
                    if (debug) {llvm::errs()<<"Find a null set var:\t"<<result<<"\n";}
                    return true;
                }
            }

            return false;
        }



    void save_result(std::string caller_name){
        if (!Result.empty())
        {
            if(debug)llvm::errs()<<"Result Size:\t"<<Result.size()<<"\n";
            std::string result = "";
            for (int i=0;i<Result.size();i++)
            {
                result += Result[i];
                if(debug)llvm::errs()<<Result[i]<<"\n";
            }
            std::ofstream out;
            out.open(output_path,std::ios::app);
            out<<result;
            if(debug)llvm::errs()<<result<<"\n";
            out.close();
        }
        visited.insert(caller_name);
        std::ofstream out;
        out.open(visited_path, std::ios::app);
        out<<caller_name + "\n";
        out.close();

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
                if (debug) {llvm::errs()<<current_funcname<<"\t";}

                
                if (visited.find(current_funcname) != visited.end()) return true;
            
                // Check wheter current function in call graph.
                //std::set<std::string>::iterator iter = FreeFuncs.find(current_funcname);
                //if(iter == FreeFuncs.end())
                 //   return true;
                
                auto funcBody = FD->getBody();
                if(!funcBody)return true;
                if(debug){llvm::errs()<<"Current funcname:\t"<<current_funcname<<"\n";}

                std::set<std::pair<int, std::string>> var_name_set;
                std::string current_callee;
                std::map<std::string,std::string> param_func_map;
        

                // Iterate with all AST nodes.
                Result.clear();
                StmtWorkStack workstack;
                workstack.push(funcBody);
                bool flag = false;
                while(!workstack.empty()){
                    auto current_stmt = workstack.pop();
                    if (auto callExpr = dyn_cast<CallExpr>(current_stmt))
                    {  
                       
                        CheckCallExpr(callExpr,var_name_set,param_func_map);
                    }
                    else if (auto binaryOperator = dyn_cast<BinaryOperator>(current_stmt))
                    {

                        CheckBinaryOperator(binaryOperator,var_name_set,param_func_map);
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
                save_result(current_funcname);
                return true;
            }
        }

    };


    class NullCheckConsumer : public ASTConsumer {
        FindFunctionVisitor Visitor;
    public:
        explicit NullCheckConsumer(ASTContext* Context) {}

        virtual void HandleTranslationUnit(clang::ASTContext &Context) {
            Visitor.TraverseDecl(Context.getTranslationUnitDecl());
        }
    };

    class FreeNullCheckAction : public PluginASTAction {
    protected:
        std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                       llvm::StringRef) override {
            return std::make_unique<NullCheckConsumer>(&CI.getASTContext());
        }

        bool ParseArgs(const CompilerInstance &CI,
                       const std::vector<std::string> &args) override {
            
            if(args.size() <3)
            {
                llvm::errs()<< "Lack input, output and visited path!\n";
                return false;
            }
            input_path = args[0];
            output_path = args[1];
            visited_path = args[2];
            bool ret = ReadFreeFuncs();
            if(!ret) return false;
            ret = read_visited();
            if(!ret) return false;
            return true;
        }
    };
}


static FrontendPluginRegistry::Add<FreeNullCheckAction>
        X("free-check", "Point the dataflow of memory functions.");
