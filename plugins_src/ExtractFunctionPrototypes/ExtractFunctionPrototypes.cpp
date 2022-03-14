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
#include <set>
using namespace clang;

std::string log_path;
std::string indirect_call_path;
template<class Data>
class FIFOWorkList {
    typedef std::set<Data> DataSet;
    typedef std::deque<Data> DataDeque;
public:
    FIFOWorkList() {}

    ~FIFOWorkList() {}

    inline bool empty() const {
        return data_list.empty();
    }

    inline bool find(Data data) const {
        return (data_set.find(data) == data_set.end() ? false : true);
    }

    inline bool push(Data data) {
        if (data_set.find(data) == data_set.end()) {
            data_list.push_back(data);
            data_set.insert(data);
            return true;
        }
        else
            return false;
    }

    inline Data pop() {
        assert(!empty() && "work list is empty");
        Data data = data_list.front();
        data_list.pop_front();
        data_set.erase(data);
        return data;
    }

    inline void clear() {
        data_list.clear();
        data_set.clear();
    }

private:
    DataSet data_set;	///< store all data in the work list.
    DataDeque data_list;	///< work list using std::vector.
};

void trim(std::string &s) 
{
    if (s.empty()) 
    {
        return ;
    }
    s.erase(0,s.find_first_not_of(" "));
    s.erase(s.find_last_not_of(" ") + 1);
}


std::string get_last_word(const std::string s)
{
    std::string new_string ="";
    if(s.find(" *") != std::string::npos)
    {
        int position = s.find(" *");
        for(int i=0;i<position;i++)
        {
            new_string += s[i];
        }

    }
    else{
        new_string =s;
    }
    
    if(new_string.find(' ') != std::string::npos)
    {
        int position = new_string.find_last_of(' ');
        std::string temp = "";
        for(int i=position+1; i<new_string.length();i++)
        {
            temp += new_string[i];
        }
        return temp;
    }
    else{
        return new_string;
    }
}

namespace {

    class PrintFunctionVisitor : public clang::RecursiveASTVisitor<PrintFunctionVisitor> {
        CompilerInstance &Instance;
        std::set<std::string> ParsedTemplates;

    public:
        PrintFunctionVisitor(CompilerInstance &Instance, std::set<std::string> ParsedTemplates)
                : Instance(Instance), ParsedTemplates(std::move(ParsedTemplates)) {}

        typedef FIFOWorkList<const Stmt*> StmtWorkList;

        std::ostringstream PrintCalleeDecl(const FunctionDecl *FD) {
            std::ostringstream ostring;
            ostring<< "\t{\"return_type\": \"" << QualType(FD->getReturnType().getTypePtr()->getUnqualifiedDesugaredType(),0).getAsString() <<"\""
                    << ", \"funcname\": \"" << FD->getNameAsString() <<"\""
                    << ", \"params\": \"";
            for (auto item : FD->parameters()){
                ostring << QualType(item->getOriginalType().getTypePtr()->getUnqualifiedDesugaredType(), 0).getAsString()
                << "@" << item->getNameAsString()  << ",";
            }
            ostring << "\"}\n";
            return ostring;
        }



        template<typename T> std::ostringstream PrintIndirectDecl(const T *FD){
            std::ostringstream ostring;
            std::string funcname= FD->getNameAsString();
            QualType type = FD->getType();
            std::string type_string = type.getAsString();
            std::string ret_type = "";
            std::string params = "";
            int i=0;
            for(;i<type_string.length();i++)
            {
                if(type_string[i] == '(')
                    break;
                ret_type += type_string[i];
            }
            for(;i<type_string.length();i++)
            {
                if(type_string[i] == ')')
                    break;
            }
            i++;
            for(i++;i<type_string.length();i++)
            {
                params += type_string[i];
            }

            std::vector<std::string> param_type;
            std::vector<std::string> param_name;
            std::string temp;
            for(i = 0;i < params.length(); i++)
            {
                
                if(params[i] == ',' or params[i] == ')')
                {
                    trim(temp);
                    param_type.push_back(temp);
                    //if(temp.find('*') != std::string::npos)
                    //{
                     //   param_name.push_back("ptr");
                   // }
                    //else
                    //{
                     //   param_name.push_back(temp);
                    //}
                    param_name.push_back(get_last_word(temp));
                    temp = "";
                    continue;
                }
                temp += params[i];
            }


            ostring<< "\t{\"return_type\": \"" << ret_type << "\""
                    << ", \"funcname\": \""<<funcname<< "\""
                    << ", \"params\": \"";
            for(i = 0;i < param_type.size(); i++)
            {
                ostring<< param_type[i] << "@" << param_name[i] <<",";
            }
            ostring << "\"}\n";


            return ostring;
        }

        void log_indirect_call(std::string str){
            std::ofstream file;
            file.open(indirect_call_path,std::ios::app);
            if(file.fail())
            {
                llvm::errs()<<"log_indirect_call can't open!";
            }
            file<<str;
            file.close();
        }

        bool VisitFunctionDecl(FunctionDecl* FD)  {
            clang::SourceManager &SourceManager = Instance.getSourceManager();
            
            if (FD) {
                FD = FD->getDefinition() == nullptr ? FD : FD->getDefinition();
                if (!FD->isThisDeclarationADefinition()) return true;

                std::ostringstream ostring;
                SourceLocation begin1 = FD->getSourceRange().getBegin();
                SourceRange sr = FD->getSourceRange();
                PresumedLoc begin = SourceManager.getPresumedLoc(sr.getBegin());
                PresumedLoc end = SourceManager.getPresumedLoc(sr.getEnd());

                ostring<< "{\"return_type\": \""<< QualType(FD->getReturnType().getTypePtr()->getUnqualifiedDesugaredType(),0).getAsString() <<"\""
                << ", \"funcname\": \"" << FD->getQualifiedNameAsString() <<"\""
                << ", \"params\": \"";
                for (auto item : FD->parameters()){
                    ostring << QualType(item->getOriginalType().getTypePtr()->getUnqualifiedDesugaredType(), 0).getAsString()
                    << "@" << item->getNameAsString()  << ",";
                }
                ostring << "\""
                << ", \"file\" :\"" << begin.getFilename() << "\""
                << ", \"begin\": [" << begin.getLine() << ", " << begin.getColumn() << "]"
                << ", \"end\": [" <<end.getLine() << ", " << end.getColumn() << "]"
                << "}" << "\n";
                auto funcBody = FD->getBody();
                if(!funcBody)return true;
                StmtWorkList worklist;
                worklist.push(funcBody);
                    
                while (!worklist.empty()) {
                    auto currentStmt = worklist.pop();
                    if (auto callExpr = dyn_cast<CallExpr>(currentStmt)) {
                        auto CD = callExpr->getCalleeDecl();
                        if (!CD) continue;
                        if (auto calleeFD = dyn_cast<FunctionDecl>(CD))
                            {
                                std::ostringstream temp_stream = PrintCalleeDecl(calleeFD);
                                ostring << temp_stream.str();
                            }
                        else if(auto calleeFD = dyn_cast<VarDecl>(CD )){
                            std::ostringstream temp_stream = PrintIndirectDecl<VarDecl>(calleeFD);
                            ostring << temp_stream.str();
                            log_indirect_call(temp_stream.str());
                        }
                        else if(auto calleeFD = dyn_cast<ValueDecl>(CD))
                        {
                            std::ostringstream temp_stream = PrintIndirectDecl<ValueDecl>(calleeFD);
                            ostring << temp_stream.str();
                            log_indirect_call(temp_stream.str());
                        }
                    }
                    for (auto stmt : currentStmt->children())
                        if(stmt)worklist.push(stmt);
                }
                std::string result = ostring.str();
                std::ofstream file;
                file.open(log_path,std::ios::app);
                file<<result;
                file.close();
                //std::cout << result;
            }
            

            return true;
        }

    };

    class PrintFunctionConsumer  : public ASTConsumer {
        PrintFunctionVisitor Visitor;
    public:
        explicit PrintFunctionConsumer(CompilerInstance &Instance, std::set<std::string> ParsedTemplates)
        :Visitor(Instance,ParsedTemplates) {}

        virtual void HandleTranslationUnit(clang::ASTContext &Context) {
            //llvm::errs()<<"HandleTranslationUnit!\n";
            Visitor.TraverseDecl(Context.getTranslationUnitDecl());
        }
    };

    class PrintFunctionNamesAction : public PluginASTAction {
        std::set<std::string> ParsedTemplates;
    protected:
        std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
                                                       llvm::StringRef) override {
            return std::make_unique<PrintFunctionConsumer>(CI, ParsedTemplates);
        }

        bool ParseArgs(const CompilerInstance &CI,
                       const std::vector<std::string> &args) override {
            if(args.size() <2)
            {
                llvm::errs()<< "Lack log path!\n";
                return false;
            }
            log_path = args[0];
            indirect_call_path = args[1];
            return true;
        }
    };

}

static FrontendPluginRegistry::Add<PrintFunctionNamesAction>
        X("extract-funcs", "Extract function prototypes from c/c++ files.");
