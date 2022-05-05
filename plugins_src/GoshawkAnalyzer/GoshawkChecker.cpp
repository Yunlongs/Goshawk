//=== MemMisuseProChecker.cpp - A Double Free and Use After Free checker -------------------*- C++ -*--//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception



#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/AST/Attr.h"
#include "clang/AST/ParentMap.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Lex/Lexer.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/BugReporter/CommonBugCategories.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramStateTrait.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SymbolManager.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "MemFuncsIdentification.h"
#include "clang/StaticAnalyzer/Frontend/CheckerRegistry.h"
#include <climits>
#include <utility>
#include <fstream>


using namespace clang;
using namespace ento;

#define DebugMode true // if true, to dump some variables' state.

#define AnalyzeMode "Customized"

namespace{

    class MemSymState{
        enum Kind{
            Allocated,
            Released
        };

        const Stmt* S;
        Kind K;

        MemSymState(const Stmt* S, Kind K):S(S),K(K){}

    public:
        bool isAllocated()const {return K==Allocated;}
        bool isReleased()const { return K==Released;}
        const Stmt *getStmt() const { return S; }
        bool operator==(const MemSymState &X) const{
            return X.K == K && X.S == S;
        }
        static MemSymState getAllocated(const Stmt* S){
            return MemSymState(S,Allocated);
        }
        static MemSymState getReleased(const Stmt *S){
            return MemSymState(S, Released);
        }
        void Profile(llvm::FoldingSetNodeID &ID) const {
            ID.AddInteger(K);
            ID.AddPointer(S);
        }
        LLVM_DUMP_METHOD void dump(raw_ostream &OS) const {
            switch (K) {
        #define CASE(ID) case ID: OS << #ID; break;
            CASE(Allocated)
            CASE(Released)
            }
        }
    };

    std::string getFunctionName(const Decl* FD){
        if (auto fd = dyn_cast<FunctionDecl>(FD))
            return fd->getNameAsString();
        else
        if (auto fd = dyn_cast<VarDecl>(FD))
            return fd->getNameAsString();
        else
        if (auto fd = dyn_cast<ValueDecl>(FD))
            return fd->getNameAsString();
    }

    // If the Function declaration we can get, then we get this function's name;
    // In some case, such as function pointer:
    //     dev->free(buf);
    // we can't get where the 'free' pointer points to, so we get this pointer's name.
    std::string getCallExprName(const CallExpr* CE, CheckerContext &C){
        std::string func_name = "";
        const FunctionDecl *FD = C.getCalleeDecl(CE);
        if (FD)
        {
            IdentifierInfo* II = FD->getIdentifier();
            if(!II)
            return "";
            func_name = II->getName().str();
        }
        else
        {
            const Decl *D  = CE->getCalleeDecl();
            if (!D)
                return "";
            func_name = getFunctionName(D);
        }
        return func_name;
    }

} // namespace end

REGISTER_MAP_WITH_PROGRAMSTATE(RegionState, SymbolRef, MemSymState)

namespace{
    long long  int extern_count = 0;
    long long  int path_number = 0;
class MemMisuseChecker: public Checker<check::DeadSymbols,eval::Call,check::PreCall,
                     check::Location,check::EndFunction> {

public:
    MemFuncsUtility *MemFunc;
    std::string Mode;
    std::string Memfunc_dir;
    std::string PathNumberFile;
    std::string ExternFile;

    MemMisuseChecker(){
    }
    
    void init(std::string MemFuncsDir, std::string PathNumberFile, std::string ExternFile){
        this->Memfunc_dir = MemFuncsDir;
        this->PathNumberFile = PathNumberFile;
        this->ExternFile = ExternFile;
        MemFunc = new MemFuncsUtility(Memfunc_dir);
        Mode = AnalyzeMode;
    }


    ~MemMisuseChecker(){
        delete MemFunc;
        write_number_line(extern_count,ExternFile);
        write_number_line(path_number+1,PathNumberFile);
    }
    void checkPreCall(const CallEvent &Call, CheckerContext &C) const;
    //void checkPostStmt(const CallExpr *CE, CheckerContext &C) const;
    bool evalCall(const CallEvent &Call, CheckerContext &C) const;
    void checkDeadSymbols(SymbolReaper &SymReaper, CheckerContext &C) const;
    //void checkPreStmt(const ReturnStmt *S, CheckerContext &C) const;
    void checkLocation(SVal l, bool isLoad, const Stmt *S,CheckerContext &C) const;
    void checkEndFunction(const ReturnStmt *RS,CheckerContext &C )const;

private:
    mutable std::unique_ptr<BugType> BT_DoubleFree;
    mutable std::unique_ptr<BugType> BT_UseAfterFree;

    DefinedOrUnknownSVal getHeapSymbolVal(const Expr *E, QualType T, SymbolManager &SymMgr,
                                      const LocationContext *LCtx,MemRegionManager &MemMgr,
                                      unsigned VisitCount)const;
    const FieldRegion* FindFieldRegion(const SubRegion* R, QualType Ty,std::string member_name, 
                                CheckerContext &C, const CallExpr* CE,unsigned &Count,ProgramStateRef &State)const;
    std::vector<std::string> ParserMemberName(std::string member_names)const;
    ProgramStateRef CreateExprHeapSymbolNormal(CheckerContext &C, const Expr *E, ProgramStateRef State) const;
    ProgramStateRef ModelMallocNormal(CheckerContext &C, const CallEvent &Call, ProgramStateRef State) const;
    ProgramStateRef ModelMallocCustomized(CheckerContext &C, const CallEvent &Call, struct MallocEntry entry, ProgramStateRef State) const;
    ProgramStateRef ModelReallocMem(CheckerContext &C, const CallEvent &Call, ProgramStateRef State)const;
    ProgramStateRef CreateHeapSymValForFree(CheckerContext &C, const Expr *ArgExpr, ProgramStateRef State) const;
    ProgramStateRef ModelFreeNormal(CheckerContext &C, const CallEvent &Call, ProgramStateRef State) const;
    ProgramStateRef ModelFreeCustomized(CheckerContext &C, const CallEvent &Call, struct FreeEntry entry, ProgramStateRef State) const;


    bool checkUseAfterFree(SymbolRef Sym, CheckerContext &C,const Stmt *S)const ;
    bool isReleased(SymbolRef Sym, CheckerContext &C)const;

    void ReportUseAfterFree(CheckerContext &C, SourceRange Range,SymbolRef Sym)const;
    void ReportDoubleFree(CheckerContext &C, SourceRange Range,SymbolRef Sym)const;
    void printState(raw_ostream &Out, ProgramStateRef State,
                const char *NL, const char *Sep) const override;
};
} // namespace end


namespace{
class MemBugVisitor final : public BugReporterVisitor{
protected:
    SymbolRef Sym;

public:
    MemBugVisitor(SymbolRef S):Sym(S) {}
    static void *getTag() {
        static int Tag = 0;
        return &Tag;
    }
    void Profile(llvm::FoldingSetNodeID &ID) const override {
        ID.AddPointer(getTag());
        ID.AddPointer(Sym);
    }
    PathDiagnosticPieceRef VisitNode(const ExplodedNode *N,BugReporterContext &BRC,
                                   PathSensitiveBugReport &BR) override;
    PathDiagnosticPieceRef getEndPath(BugReporterContext &BRC,const ExplodedNode *EndPathNode,
                                    PathSensitiveBugReport &BR) override {
        PathDiagnosticLocation L = BR.getLocation();
        // Do not add the statement itself as a range in case of leak.
        return std::make_shared<PathDiagnosticEventPiece>(L, BR.getDescription(),false);
    }
    bool isAllocated(const MemSymState *RSCurr, const MemSymState* RSPrev, const Stmt* S){
        return S &&( RSCurr &&RSCurr->isAllocated() && (!RSPrev || !RSPrev->isAllocated()));
    }
    bool isReleased(const MemSymState* RSCurr, const MemSymState* RSPrev, const Stmt* S){
        return S && (RSCurr && RSCurr->isReleased() && (!RSPrev || !RSPrev->isReleased()));
    }

private:
  class StackHintGeneratorForReallocationFailed
      : public StackHintGeneratorForSymbol {
    public:
        StackHintGeneratorForReallocationFailed(SymbolRef S, StringRef M)
            : StackHintGeneratorForSymbol(S, M) {}

        std::string getMessageForArg(const Expr *ArgE, unsigned ArgIndex) override {
            // Printed parameters start at 1, not 0.
            ++ArgIndex;

            SmallString<200> buf;
            llvm::raw_svector_ostream os(buf);

            os << "Reallocation of " << ArgIndex << llvm::getOrdinalSuffix(ArgIndex)
                << " parameter failed";

            return std::string(os.str());
        }
    };

};
} // end namespace


/*
    As the SValBuilder::getConjuredHeapSymbolVal(const Expr *E,...,) function
    will conjure a new HeapSymbolVal which type is E->getType(), not we ofter want.
    So, I wrapper the conjureSymbol() function and can specific the QualType T that
    we want.
*/
DefinedOrUnknownSVal MemMisuseChecker::getHeapSymbolVal(const Expr *E, QualType T, SymbolManager &SymMgr,
                                      const LocationContext *LCtx,MemRegionManager &MemMgr,
                                      unsigned VisitCount)const {

  assert(Loc::isLocType(T));
  assert(SymbolManager::canSymbolicate(T));
  SymbolRef sym = SymMgr.conjureSymbol(E, LCtx, T, VisitCount);
  return loc::MemRegionVal(MemMgr.getSymbolicHeapRegion(sym));
}

/*
    Here we only get the final member region, not allocate for them.
*/
const FieldRegion* FindFinalRegion(const SubRegion* R, QualType Ty,std::string member_name, 
                                CheckerContext &C)
{
    StoreManager &StoreMgr = C.getStoreManager();
    ProgramStateRef State = C.getState();
    MemRegionManager &MemMgr = StoreMgr.getRegionManager();
    if(DebugMode){llvm::errs()<<"\nCurrent member name:\t"<<member_name<<"\n";}
    if(Ty->isPointerType())
    {
        Ty = Ty->getAs<PointerType>()->getPointeeType();
    }

    if (const RecordType* RT = Ty->getAsStructureType())
    {
        const RecordDecl* RD = RT->getDecl()->getDefinition();
        if(!RD)
        {
            if(DebugMode) {llvm::errs()<<"RecordType :\t";RT->dump();llvm::errs()<<"\n";}
            if(DebugMode) {llvm::errs()<<"RecordDecl :\t";RT->getDecl()->dump();llvm::errs()<<"\n";}
            return nullptr;
        }  
            
        for (const FieldDecl* FieldD : RD->fields())
        {
            std::string FieldName = FieldD->getNameAsString();
            if(FieldName == member_name)
            {
                const FieldRegion* FR = MemMgr.getFieldRegion(FieldD, R);
                return FR;
            } 
        }
    }
    return nullptr;
}

/*
    Here we will Traversal the structure member of Region R,
    and get the member structure HeapSymbolVal allocated.
*/
const FieldRegion* MemMisuseChecker::FindFieldRegion(const SubRegion* R, QualType Ty,std::string member_name, 
                                CheckerContext &C, const CallExpr* CE,unsigned &Count,ProgramStateRef &State)const
{
  const LocationContext *LCtx = C.getPredecessor()->getLocationContext();
  StoreManager &StoreMgr = C.getStoreManager();
  SymbolManager &SymMgr = C.getSymbolManager();
  MemRegionManager &MemMgr = StoreMgr.getRegionManager();
  bool Prev_PointerTy = false;
  if (Ty->isPointerType())
  {
    Ty = Ty->getAs<PointerType>()->getPointeeType();
    Prev_PointerTy = true;
  }
  if (Ty->isPointerType()) // for **
  {
    Ty = Ty->getAs<PointerType>()->getPointeeType();
    Prev_PointerTy = true;
  }

  if (const RecordType* RT = Ty->getAsStructureType())
  {
    const RecordDecl* RD = RT->getDecl()->getDefinition();
    if (!RD)
      return nullptr;
    
    /*
      Traverse the FieldDecl of this structure definition.
      Find the Field corresponding to member_name.
    */
    for (const FieldDecl* Field : RD->fields())
    {
      if (Field->getNameAsString() == member_name)
      {
        const FieldRegion* FR = MemMgr.getFieldRegion(Field,R);
        if (DebugMode) {llvm::errs()<<"Current member: "<<member_name<<"\t FieldRegion \t";FR->dump(); llvm::errs()<<"\n";}
        SVal FRVal = StoreMgr.getBinding(State->getStore(), loc::MemRegionVal(FR));
        if(DebugMode) {llvm::errs()<<"Get FR binding Val:\t";FRVal.dump();llvm::errs()<<"\n";}
        if (!FRVal.isUndef())
            return FR;
        QualType T = Field->getType()->getCanonicalTypeUnqualified();
        if (T->isAnyPointerType() && !Prev_PointerTy)
        {
          if(DebugMode){loc::MemRegionVal(FR).dump();llvm::errs()<<"\n";}
          DefinedSVal MemVal = getHeapSymbolVal(CE, T, SymMgr, LCtx, MemMgr, Count++)    
                                .castAs<DefinedSVal>();
          State = State->bindDefaultInitial(MemVal, UndefinedVal(), LCtx);
          State = State->bindLoc(loc::MemRegionVal(FR), MemVal, LCtx);
          SymbolRef Sym = MemVal.getAsSymbol();
          assert(Sym);
          State = State->set<RegionState>(Sym, MemSymState::getAllocated(CE));
        }
        else if(T->isAnyPointerType() && Prev_PointerTy)
        {
          if(DebugMode) {llvm::errs()<<"loc::MemRegionVal(FR):\t";loc::MemRegionVal(FR).dump();llvm::errs()<<"\n";}
          SVal val = StoreMgr.getBinding(State->getStore(),loc::MemRegionVal(R));
          if(DebugMode) {llvm::errs()<<"getBinding Val:\t";val.dump();llvm::errs()<<"\n";}
          if (val.getAs<nonloc::LazyCompoundVal>())
          {
            if(DebugMode) {llvm::errs()<<" LazyCompoundVal Branch.\n";}
            DefinedSVal MemVal = getHeapSymbolVal(CE, T, SymMgr, LCtx, MemMgr, Count++)    
                                .castAs<DefinedSVal>();
            if (DebugMode) {llvm::errs()<< "New Heap Val:\t";MemVal.dump();llvm::errs()<<"\n";}
            State = State->bindDefaultInitial(MemVal, UndefinedVal(), LCtx);
            if (DebugMode) {llvm::errs()<< "bindDefaultInitial\n\n";}
            State = State->bindLoc(loc::MemRegionVal(FR), MemVal, LCtx);
            SymbolRef Sym = MemVal.getAsSymbol();
            assert(Sym);
            State = State->set<RegionState>(Sym, MemSymState::getAllocated(CE));
          }
          else{
            if(DebugMode) {llvm::errs()<<"\n Not LazyCompoundVal Branch.\n";}
            if (!val.getAsRegion())
                return nullptr;
            const SubRegion* SubR = val.getAsRegion()->getAs<SubRegion>();
            if (!SubR)
              return nullptr;
            const FieldRegion* NewFR = MemMgr.getFieldRegion(Field,SubR);
            DefinedSVal MemVal = getHeapSymbolVal(CE, T, SymMgr, LCtx, MemMgr, Count++)    
                                .castAs<DefinedSVal>();
            if (DebugMode) {llvm::errs()<< "New Heap Val:\t";MemVal.dump();llvm::errs()<<"\n\n";}
            State = State->bindDefaultInitial(MemVal, UndefinedVal(), LCtx);
            State = State->bindLoc(loc::MemRegionVal(NewFR), MemVal, LCtx);
            SymbolRef Sym = MemVal.getAsSymbol();
            assert(Sym);
            State = State->set<RegionState>(Sym, MemSymState::getAllocated(CE));
            return NewFR;
          }
        }
        return FR;
      }
    }
  }
  return nullptr;
}


/*
  Parse a member name "->sock.buf" to a ["sock","buf"] vector.
*/
std::vector<std::string> MemMisuseChecker::ParserMemberName(std::string member_names) const
{
  std::vector<std::string> vector;
  if (member_names.length()<=0)
    return vector;
  
  std::string new_string;
  for(size_t i = 0; i < member_names.length(); i++)
  {
    if(member_names[i] == '.')
      {
        if (new_string.length()!=0)
          vector.push_back(new_string);
        new_string = "";
        continue;
      }
    if(member_names[i] == '-')
      {
        if (new_string.length()!=0)
          vector.push_back(new_string);
        new_string = "";
        i++;continue;
      }
    new_string += member_names[i];
  }
  vector.push_back(new_string);
  return vector;
}

/*
    This function will create a HeapSymolVal to the return value
    for Normal Allocation function.

    Note: Normal Allocation function means return a new heap only
    by return value, like 'malloc'.
*/
ProgramStateRef MemMisuseChecker::ModelMallocNormal(CheckerContext &C, const CallEvent &Call, ProgramStateRef State) const{



    const Expr* expr = Call.getOriginExpr();
    if(!expr)
        return nullptr;
    const CallExpr *CE = dyn_cast<CallExpr>(expr);
    if(!CE)
        return nullptr;

    unsigned Count = C.blockCount();
    SValBuilder &svalBuilder = C.getSValBuilder();
    const LocationContext *LCtx = C.getPredecessor()->getLocationContext();
    DefinedSVal RetVal = svalBuilder.getConjuredHeapSymbolVal(CE, LCtx, Count)    
      .castAs<DefinedSVal>();

      // Fill the region with the initialization value.
    State = State->bindDefaultInitial(RetVal, UndefinedVal(), LCtx);
    State = State->BindExpr(CE, C.getLocationContext(), RetVal);
  
    if(!RetVal.getAs<Loc>())
        return nullptr;

    SymbolRef Sym = RetVal.getAsLocSymbol();
    assert(Sym);
    return State->set<RegionState>(Sym,MemSymState::getAllocated(CE));
}

/*
    This function will model the heap allocation behaviors for Customized
    Allocation functions.

    Firstly, we create a HeapSymbolVal to reutrn value.
    Secondly, we create HeapSymbolVals for the member structures of return value.
    Thirdly, we create HeapSymbolVals to arguments.
    Finally, we create HeapSymbolVals for the member structures of arguments.

    Note: Customized Allocation function means it sometimes is a wrapper function
    of Normal Allocation function. And it often initializes the structure before 
    return the allocated heap.
*/
ProgramStateRef MemMisuseChecker::ModelMallocCustomized(CheckerContext &C, const CallEvent &Call, 
                                        struct MallocEntry entry, ProgramStateRef State) const
{
    // Get State and Managers.
    unsigned Count = C.blockCount();
    SValBuilder &svalBuilder = C.getSValBuilder();
    const LocationContext *LCtx = C.getPredecessor()->getLocationContext();
    
    // Check if this Call is valid.
    const Expr* expr = Call.getOriginExpr();
    if(!expr)
        return nullptr;
    const CallExpr* CE = dyn_cast<CallExpr>(expr);
    if(!CE)
        return nullptr;
    const FunctionDecl* FD = CE->getDirectCallee();
    if(!FD)
        return nullptr;
    int arg_num = CE->getNumArgs();

    if (DebugMode){llvm::errs()<<"ModelMallocCustomized!\n\n-------------\n\n";}

    if (entry.MemberVector.size() !=0)
    {
        // Create HeapSymbolVal to return value.
        DefinedSVal RetVal = svalBuilder.getConjuredHeapSymbolVal(CE, LCtx, Count++)    
        .castAs<DefinedSVal>();
        auto RetRegion = RetVal.getAsRegion()->getAs<SubRegion>();
        if(!RetVal.getAs<Loc>() || !RetRegion)
        return nullptr;
        if(DebugMode){llvm::errs()<<"\nRetVal: ";RetVal.dump();llvm::errs()<<"\n";}
        State = State->bindDefaultInitial(RetVal, UndefinedVal(), LCtx);
        SymbolRef Sym = RetVal.getAsLocSymbol();
        assert(Sym);
        State = State->set<RegionState>(Sym, MemSymState::getAllocated(CE));


        QualType RetTy = CE->getCallReturnType(C.getASTContext())->getCanonicalTypeUnqualified();

        /*
            Process the structure member of return value. like "->sock.buf".
        */
        for(std::string member_name : entry.MemberVector)
        {
            std::vector<std::string> member_vector = ParserMemberName(member_name);
            auto R = RetRegion;
            auto Ty = RetTy;
            for (std::string name : member_vector)
            {
                const FieldRegion * FR = FindFieldRegion(R, Ty, name, C, CE, Count, State);
                if (!FR)
                    break;
                Ty = FR->getValueType()->getCanonicalTypeUnqualified();
                R = FR->getAs<SubRegion>();
            }
        }
        State = State->BindExpr(CE,LCtx,RetVal);
    }



    /*
      Process the direct assigned parameters. like "*poll = malloc(..);"
    */
    
    for (std::pair<int,std::string> pair : entry.ParameterVector)
    {
        // First check if the provide information is corresponding to this Call.
        int ArgIndex = pair.first;
        std::string ParamName = pair.second;
        if (ArgIndex >= arg_num)
            continue;
        const Expr *Arg_expr = CE->getArg(ArgIndex);
        if (FD->getParamDecl(ArgIndex)->getNameAsString() != ParamName)
            continue;

        if(DebugMode) {llvm::errs()<<"\n\n Process Parameter:\t"<<ParamName<<"\n";}
        // Create a New HeapSymbolVal for this argument.
        SVal ArgVal = C.getSVal(Arg_expr);
        if (DebugMode){llvm::errs()<<"\nArgVal dump:\t";ArgVal.dump();llvm::errs()<<"\n";}
        DefinedSVal ArgHeapVal = svalBuilder.getConjuredHeapSymbolVal(Arg_expr, LCtx, Count++)    
                        .castAs<DefinedSVal>();
        State = State->bindDefaultInitial(ArgHeapVal, UndefinedVal(), LCtx);
        State = State->bindLoc(ArgVal, ArgHeapVal, LCtx);
        SymbolRef Sym = ArgHeapVal.getAsSymbol();
        assert(Sym);
        State = State->set<RegionState>(Sym, MemSymState::getAllocated(Arg_expr));
    }
    


    /*
      Process the parameter's structure member. like "arg1 ->sock.buf"
    */
    std::vector<std::pair<int,std::string>> ParamMemberVector;
    for (std::pair<int, std::string> pair : entry.ParamMemberVector)
    {
        // First check if the provide information is corresponding to this Call.
        int MemberIndex = pair.first;
        std::string ParamMemberName = pair.second;
        if (MemberIndex >= arg_num)
            continue;
        if(DebugMode) {llvm::errs()<<"\n\n Process Parameter Member:\t"<<ParamMemberName<<"\n";}
        const Expr* ArgMem_expr = CE->getArg(MemberIndex);
        SVal ArgMemVal = C.getSVal(ArgMem_expr);
        if (DebugMode){llvm::errs()<<"\nArgMemVal dump:\t";ArgMemVal.dump();llvm::errs()<<"\n";}
        if (!ArgMemVal.getAsRegion())
            return nullptr;
        auto R = ArgMemVal.getAsRegion()->getAs<SubRegion>();
        auto Ty = ArgMem_expr->getType()->getCanonicalTypeUnqualified();
        auto member_vector = ParserMemberName(ParamMemberName);
        for (std::string name : member_vector)
        {
            const FieldRegion * FR = FindFieldRegion(R, Ty, name, C, CE, Count, State);
            if (!FR)
                break;
            Ty = FR->getValueType()->getCanonicalTypeUnqualified();
            R = FR->getAs<SubRegion>();
        }        
    }
    return State;
}


ProgramStateRef MemMisuseChecker::ModelFreeNormal(CheckerContext &C, const CallEvent &Call, ProgramStateRef State)const{


    // Check this Call is valid.
    const Expr* expr = Call.getOriginExpr();
    if(!expr)
        return nullptr;
    const CallExpr *CE = dyn_cast<CallExpr>(expr);
    if(!CE)
        return nullptr;
    if(CE->getNumArgs()<1)
        return nullptr;
    const Expr* freed_arg =  CE->getArg(0);
    if (!freed_arg)
        return nullptr;

    // Check Wether this Expr contains a prefix '&', if contains directly return.
    // In this case, it will also release it's baseRegion.
    // So we need to assign a heap region to this member variable.
    /*Example: 
        put_device(&device->dev); //it release the device memory, not device->dev.
        device->member = some // cause UAF!
    */
    
    if (auto unaryOperator = dyn_cast<UnaryOperator>(freed_arg->IgnoreCasts()))
    {
        std::string name = unaryOperator->getOpcodeStr(unaryOperator->getOpcode()).str();
        if (name == "&")
        {
            return nullptr;       
        }
    }
    
    const LocationContext *LCtx = C.getPredecessor()->getLocationContext();
    SVal ArgVal = State->getSVal(freed_arg,LCtx);
    if (!ArgVal.getAs<DefinedOrUnknownSVal>())
        return nullptr;
    const MemRegion *R = ArgVal.getAsRegion();
    if (!R)
        return nullptr;
    if(DebugMode) {R->dump(); llvm::errs()<<"\n";} 

    R = R->StripCasts();
    if(DebugMode) {llvm::errs()<<"\n After Strip:\t";R->dump(); llvm::errs()<<"\n";} 

    const SymbolicRegion *SrBase = dyn_cast<SymbolicRegion>(R->getBaseRegion());
    if (!SrBase)
        return nullptr;
    if(DebugMode) {llvm::errs()<<"\n Get BaseRegion:\t";SrBase->dump(); llvm::errs()<<"\n";} 


    SymbolRef SymBase = SrBase->getSymbol();
    if(DebugMode) {llvm::errs()<<"\n Get Base Sym:\t";SymBase->dump(); llvm::errs()<<"\n";} 
    const MemSymState *RsBase = State->get<RegionState>(SymBase);
    if (!RsBase)
    {
        return State->set<RegionState>(SymBase,MemSymState::getReleased(freed_arg));
    }

    if(RsBase->isReleased())
    {
        ReportDoubleFree(C,CE->getSourceRange(),SymBase);
        return nullptr;
    }
    return State->set<RegionState>(SymBase,MemSymState::getReleased(freed_arg));
}


ProgramStateRef MemMisuseChecker::ModelFreeCustomized(CheckerContext &C, const CallEvent &Call, 
                                    struct FreeEntry entry, ProgramStateRef State)const{
    // Check this Call is valid.
    StoreManager &StoreMgr = C.getStoreManager();
    const Expr* expr = Call.getOriginExpr();
    if(!expr)
        return nullptr;
    const CallExpr *CE = dyn_cast<CallExpr>(expr);
    if(!CE)
        return nullptr;
    if(CE->getNumArgs()<1)
        return nullptr;    
    const FunctionDecl* FD = CE->getDirectCallee();
    if(!FD)
        return nullptr;
    int ParamNum = FD->getNumParams();

    if(DebugMode) {llvm::errs()<<"\n\n ModelFreeCustomized!-------------------\n\n";}

    /*
        Process the Parameter first.
    */
    std::vector<int> ParamIndex = entry.ParamIndex;
    for (int index : ParamIndex)
    {
        if (index >= ParamNum)
            return nullptr;
        std::string param_name = FD->getParamDecl(index)->getNameAsString();

        SVal ArgVal = Call.getArgSVal(index);
        if (!ArgVal.getAs<DefinedOrUnknownSVal>())
            return nullptr;
        const MemRegion *R = ArgVal.getAsRegion();
        if (!R)
            break;
        if (DebugMode){llvm::errs()<<"Param Name: "<<param_name<<"\t MemRegion:\t";R->dump();llvm::errs()<<"\n";}
        R = R->StripCasts();
        if (DebugMode){llvm::errs()<<"After Strip:\t";R->dump();llvm::errs()<<"\n";}
        const SymbolicRegion *SrBase = dyn_cast<SymbolicRegion>(R);
        if (!SrBase)
            return nullptr;
        if (DebugMode){llvm::errs()<<"After dyn_cast<SymbolicRegion>:\t";SrBase->dump();llvm::errs()<<"\n";}

        SymbolRef SymBase = SrBase->getSymbol();
        if (DebugMode){llvm::errs()<<"Get the SymbolRef:\t";SymBase->dump();llvm::errs()<<"\n";}
        const MemSymState *RsBase = State->get<RegionState>(SymBase);
        if(!RsBase)
            break;
        if(RsBase->isReleased())
        {
            ReportDoubleFree(C,CE->getSourceRange(),SymBase);
            return nullptr;
        }
        State = State->set<RegionState>(SymBase,MemSymState::getReleased(Call.getArgExpr(index)));
        break;
    
    }

    /*
        Process the Structure Member.
    */
    for (int index = 0; index < ParamNum; index++)
    {
        std::string ParamName = FD->getParamDecl(index)->getNameAsString();
        for(std::string member_name : entry.ParamMeberVector)
        {
            std::vector<std::string> member_vector = ParserMemberName(member_name);
            std::string BaseName = member_vector[0];
            if (BaseName != ParamName)
                continue;

            const Expr* ArgExpr = Call.getArgExpr(index);
            SVal ArgVal = Call.getArgSVal(index);
            auto Region = ArgVal.getAsRegion();
            if (!Region)
            {
                if(DebugMode) {llvm::errs()<<"ArgRegion:\t";ArgVal.dump();llvm::errs()<<"\n";}
                return nullptr;
            }
            auto ArgRegion = ArgVal.getAsRegion()->getAs<SubRegion>();
            if (!ArgRegion)
                return nullptr;
            
            QualType ArgTy = ArgExpr->getType()->getCanonicalTypeUnqualified();
            auto R = ArgRegion;
            auto Ty = ArgTy;
            for (size_t i = 1; i < member_vector.size(); i++)// because element at 0,is basename.
            {
                const FieldRegion* FR = FindFinalRegion(R, Ty, member_vector[i], C);
                if (!FR)
                    return nullptr;
                SVal FRVal = StoreMgr.getBinding(State->getStore(),loc::MemRegionVal(FR));
                if(DebugMode) {llvm::errs()<<"FindFinalRegion: FRVal binding:\t";FRVal.dump();llvm::errs()<<"\n\n";}
                if (FRVal.isUndef() || FRVal.isZeroConstant() || !FRVal.getAsRegion())
                    R = FR;
                else if(FRVal.getAs<nonloc::LazyCompoundVal>())
                    R = FR;
                else
                    R = FRVal.getAsRegion()->getAs<SubRegion>();
                Ty = FR->getValueType()->getCanonicalTypeUnqualified();
            }
            if (DebugMode) {llvm::errs()<<"FieldRegion FR:\t";R->dump();llvm::errs()<<"\n";}
           
            const SymbolicRegion *SrBase = dyn_cast<SymbolicRegion>(R);
            if (!SrBase)
                return nullptr;
            if (DebugMode){llvm::errs()<<"After dyn_cast<SymbolicRegion>:\t";SrBase->dump();llvm::errs()<<"\n";}

            SymbolRef SymBase = SrBase->getSymbol();
            if (!SymBase)
                return nullptr;
            if (DebugMode) {llvm::errs()<<"FieldRegion SymBase:\t";SymBase->dump();llvm::errs()<<"\n";}
            const MemSymState *RsBase = State->get<RegionState>(SymBase);
            if(!RsBase)
                return nullptr;
            State = State->set<RegionState>(SymBase,MemSymState::getReleased(Call.getArgExpr(index)));
            if(RsBase->isReleased())
            {
                ReportDoubleFree(C,CE->getSourceRange(),SymBase);
                return nullptr;
            }
        }
    }
    return State;
}



ProgramStateRef MemMisuseChecker::ModelReallocMem(CheckerContext &C, const CallEvent &Call, ProgramStateRef State)const{
    if (!State)
        return nullptr;
    const Expr* expr = Call.getOriginExpr();
    if(!expr)
        return nullptr;
    const CallExpr *CE = dyn_cast<CallExpr>(expr);
    if(!CE)
        return nullptr;
    
    if(DebugMode) {llvm::errs()<<"Enter a realloc function\n";}

    // If the number of arguments less than 2, it could not be a realloc function.
    if (CE->getNumArgs()<2)
        return nullptr;
    
    const Expr *arg0Expr = CE->getArg(0);
    if(DebugMode) {llvm::errs()<<"Arg0Expr dump:";arg0Expr->dump();llvm::errs()<<"\n";}
    SVal Arg0Val = C.getSVal(arg0Expr);
    if (!Arg0Val.getAs<DefinedOrUnknownSVal>())
        return nullptr;
    QualType ptrTy = Arg0Val.getAsSymbol()->getType();

    DefinedOrUnknownSVal arg0Val = Arg0Val.castAs<DefinedOrUnknownSVal>();

    SValBuilder &svalBuilder = C.getSValBuilder();

    DefinedOrUnknownSVal PtrEQ =
        svalBuilder.evalEQ(State, arg0Val, svalBuilder.makeNullWithType(ptrTy));

    const Expr *Arg1 = CE->getArg(1);
    if(DebugMode) {llvm::errs()<<"Arg1Expr dump:";Arg1->dump();llvm::errs()<<"\n";}

    // Get the value of the size argument.
    SVal TotalSize = C.getSVal(Arg1);
    if (!TotalSize.getAs<DefinedOrUnknownSVal>())
        return nullptr;

    if (!TotalSize.getAsSymbol())
        return nullptr;
    QualType sizeTy = TotalSize.getAsSymbol()->getType();
    if (DebugMode){llvm::errs()<<"size type:\t";sizeTy.dump();llvm::errs()<<"\n";}

    // Compare the size argument to 0.
    DefinedOrUnknownSVal SizeZero =
    svalBuilder.evalEQ(State, TotalSize.castAs<DefinedOrUnknownSVal>(),
                       svalBuilder.makeIntValWithWidth(sizeTy, 0));
    if (DebugMode){llvm::errs()<<"Compare the size argument to 0.";}

    ProgramStateRef StatePtrIsNull, StatePtrNotNull;
    std::tie(StatePtrIsNull, StatePtrNotNull) = State->assume(PtrEQ);
    ProgramStateRef StateSizeIsZero, StateSizeNotZero;
    std::tie(StateSizeIsZero, StateSizeNotZero) = State->assume(SizeZero);
    // We only assume exceptional states if they are definitely true; if the
    // state is under-constrained, assume regular realloc behavior.
    bool PrtIsNull = StatePtrIsNull && !StatePtrNotNull;
    bool SizeIsZero = StateSizeIsZero && !StateSizeNotZero;

      // If the ptr is NULL and the size is not 0, the call is equivalent to
    // malloc(size).
    if (PrtIsNull && !SizeIsZero) {
        ProgramStateRef stateMalloc = ModelMallocNormal(C, Call, State);
        return stateMalloc;
    }

    // If the reallocated ptr is NULL and size is 0, this function do nothing.
    if (PrtIsNull && SizeIsZero)
        return State;
    
    // Get the from and to pointer symbols as in toPtr = realloc(fromPtr, size).
    assert(!PrtIsNull);
    SymbolRef FromPtr = arg0Val.getAsSymbol();
    SVal RetVal = C.getSVal(CE);
    SymbolRef ToPtr = RetVal.getAsSymbol();
    if (!FromPtr || !ToPtr)
        return nullptr;
    
    if (SizeIsZero)
    // If size was equal to 0, either NULL or a pointer suitable to be passed
    // to free() is returned. 
        if (ProgramStateRef stateFree =
                ModelFreeNormal(C, Call, StateSizeIsZero))
            return stateFree;

    // Normal behavior of realloc
    if (ProgramStateRef stateFree = ModelFreeNormal(C, Call, State)) {
        ProgramStateRef stateRealloc = ModelMallocNormal(C, Call,stateFree);
        if (!stateRealloc)
            return nullptr;
        return stateRealloc;
    }
    return nullptr;
}

bool MemMisuseChecker::evalCall(const CallEvent &Call, CheckerContext &C) const{
    const auto *FD = dyn_cast_or_null<FunctionDecl>(Call.getDecl());
    if (!FD || FD->getKind() != Decl::Function)
        return false;

    if (!Call.isGlobalCFunction())
        return false;
    bool hasBody = FD->hasBody();

    ProgramStateRef State = C.getState();
    std::string func_name = FD->getNameAsString();
    if(DebugMode){llvm::errs()<<"Current Function Name:\t"<<func_name<<"\n";}
   
    if (struct MallocEntry* entry = MemFunc->isMallocFunction(func_name))
    {
        if(entry->kind == Normal)
            State = ModelMallocNormal(C, Call, State);
        else if(entry->kind == Customized)
            State = ModelMallocCustomized(C, Call, *entry, State);
    }
    else if(struct FreeEntry* entry = MemFunc->isFreeFunction(func_name))
    {
        if (entry->kind == Normal)
            State = ModelFreeNormal(C, Call, State);
        else if(entry->kind == Customized)
            State = ModelFreeCustomized(C, Call, *entry, State);
        else
            return false;
    }
    else if(MemFunc->isReallocFunction(func_name))
    {
        State = ModelReallocMem(C, Call, State);
    }
    else
    {
        return false;
    }
    if (State)
        C.addTransition(State);
    bool isDifferent = C.isDifferent();
    if (!hasBody && isDifferent)
    {
        extern_count +=1;
    }
    return isDifferent;
}


void MemMisuseChecker::checkPreCall(const CallEvent &Call, CheckerContext &C) const{
    if (const AnyFunctionCall *FC = dyn_cast<AnyFunctionCall>(&Call)) {
        const FunctionDecl *FD = FC->getDecl();
        if (!FD)
            return;
        std::string func_name = FD->getNameAsString();

        if(MemFunc->isFreeFunction(func_name)) // free function will be eval at evalcall.
        {
            return; 
        }
    }


    for (unsigned I = 0, E = Call.getNumArgs(); I != E; ++I) {
    SVal ArgSVal = Call.getArgSVal(I);
    if (ArgSVal.getAs<Loc>()) {
      SymbolRef Sym = ArgSVal.getAsSymbol();
      if (!Sym)
        continue;
      if (checkUseAfterFree(Sym, C, Call.getArgExpr(I)))
        return;
    }
  }
}


bool MemMisuseChecker::isReleased(SymbolRef Sym,CheckerContext &C)const{
    ProgramStateRef State = C.getState();
    const MemSymState *memState = State->get<RegionState>(Sym);
    if(!memState)
        return false;
    return memState->isReleased();
}

bool MemMisuseChecker::checkUseAfterFree(SymbolRef Sym, CheckerContext &C,
                                      const Stmt *S) const{

  if (isReleased(Sym, C)) {
    ReportUseAfterFree(C, S->getSourceRange(), Sym);
    return true;
  }

  return false;
}

void MemMisuseChecker::checkDeadSymbols(SymbolReaper &SymReaper, CheckerContext &C) const{
    /*Future: Support Memleak check in the future.*/
    ProgramStateRef state = C.getState();
    RegionStateTy OldRS = state->get<RegionState>();
    RegionStateTy::Factory &F = state->get_context<RegionState>();

    RegionStateTy RS = OldRS;
    SmallVector<SymbolRef, 2> Errors;
    for (RegionStateTy::iterator I = RS.begin(), E = RS.end(); I != E; ++I) {
        if (SymReaper.isDead(I->first)) {
        // Remove the dead symbol from the map.
        RS = F.remove(RS, I->first);
        }
    }
}

// Check if the location is a freed symbolic region.
void MemMisuseChecker::checkLocation(SVal l, bool isLoad, const Stmt *S,
                                  CheckerContext &C) const {
  SymbolRef Sym = l.getLocSymbolInBase();
  if (Sym) {
    checkUseAfterFree(Sym, C, S);
  }
}


void MemMisuseChecker::printState(raw_ostream &Out, ProgramStateRef State,
                               const char *NL, const char *Sep) const {

  RegionStateTy RS = State->get<RegionState>();

  if (!RS.isEmpty()) {
    Out << Sep << "LoccsChecker :" << NL;
    for (RegionStateTy::iterator I = RS.begin(), E = RS.end(); I != E; ++I) {
      I.getKey()->dumpToStream(Out);
      Out << " : ";
      I.getData().dump(Out);
      Out << NL;
    }
  }
}

void MemMisuseChecker::ReportUseAfterFree(CheckerContext &C, SourceRange Range,SymbolRef Sym)const
{
    if(ExplodedNode *N = C.generateErrorNode())
    {
        if(!BT_UseAfterFree)
        {
            BT_UseAfterFree.reset(new BugType(this, "Use-after-free", categories::MemoryError));
        }
        auto R = std::make_unique<PathSensitiveBugReport>(*BT_UseAfterFree,"Use of memory after it is freed",N);
        R->markInteresting(Sym);
        R->addRange(Range);
        R->addVisitor(std::make_unique<MemBugVisitor>(Sym));
        C.emitReport(std::move(R));
    }
}

void MemMisuseChecker::ReportDoubleFree(CheckerContext &C, SourceRange Range,SymbolRef Sym) const{
    if(ExplodedNode *N = C.generateErrorNode())
    {
        if(!BT_DoubleFree)
        {
            BT_DoubleFree.reset(new BugType(this, "Double Free", categories::MemoryError));
        }
        auto R = std::make_unique<PathSensitiveBugReport>(*BT_DoubleFree,"Free a memory which is already freed",N);
        R->markInteresting(Sym);
        R->addRange(Range);
        R->addVisitor(std::make_unique<MemBugVisitor>(Sym));
        C.emitReport(std::move(R));
    }
}

void MemMisuseChecker::checkEndFunction(const ReturnStmt *RS,CheckerContext &C) const{
    /*
    std::ofstream file;
    file.open("/tmp/CSA/path_number",std::ios::app);
    file<<"1";
    file.close();
    */
   path_number += 1;
}

PathDiagnosticPieceRef MemBugVisitor::VisitNode(const ExplodedNode *N,BugReporterContext &BRC,
                                PathSensitiveBugReport &BR) {
    ProgramStateRef state = N->getState();
    ProgramStateRef statePrev = N->getFirstPred()->getState();

    const MemSymState *RSCurr = state->get<RegionState>(Sym);
    const MemSymState *RSPrev = statePrev->get<RegionState>(Sym);

    const Stmt *S = N->getStmtForDiagnostics();
    // When dealing with containers, we sometimes want to give a note
    // even if the statement is missing.
    if (!S && (!RSCurr))
        return nullptr;
    
    StringRef Msg;
    std::unique_ptr<StackHintGeneratorForSymbol> StackHint = nullptr;
    SmallString<256> Buf;
    llvm::raw_svector_ostream OS(Buf);

    if(isAllocated(RSCurr,RSPrev,S)){
        Msg = "Memory is allocated";
        StackHint = std::make_unique<StackHintGeneratorForSymbol>(Sym, "Returned allocated memory");
    } 
    else if (isReleased(RSCurr, RSPrev, S)) {
        Msg = "Memory is released";
        StackHint = std::make_unique<StackHintGeneratorForSymbol>(Sym, "Returning; memory was released");
    }

     if (Msg.empty()) {
    assert(!StackHint);
    return nullptr;
  }

    assert(StackHint);

    // Generate the extra diagnostic.
    PathDiagnosticLocation Pos;
    
    Pos = PathDiagnosticLocation(S, BRC.getSourceManager(),
                                 N->getLocationContext());
    auto P = std::make_shared<PathDiagnosticEventPiece>(Pos, Msg, true);
    if (!P)
        return nullptr;
    BR.addCallStackHint(P, std::move(StackHint));
    return P;
}

void registerMemMisuseProChecker(CheckerManager &Mgr) {
    MemMisuseChecker *Checker = Mgr.registerChecker<MemMisuseChecker>();
    std::string MemFuncsDir = std::string(Mgr.getAnalyzerOptions().getCheckerStringOption(Checker, "MemFuncsDir"));
    std::string PathNumberFile = std::string(Mgr.getAnalyzerOptions().getCheckerStringOption(Checker, "PathNumberFile"));
    std::string ExternFile = std::string(Mgr.getAnalyzerOptions().getCheckerStringOption(Checker, "ExternFile"));
    //llvm::errs()<<"This is an options:"<<MemFuncsDir<<"\n";
    Checker->init(MemFuncsDir, PathNumberFile, ExternFile);
}

bool shouldRegisterMemMisuseProChecker(const CheckerManager &mgr) {return true;}


// Register plugin!
extern "C" void clang_registerCheckers(CheckerRegistry &registry) {
    registry.addChecker(registerMemMisuseProChecker,shouldRegisterMemMisuseProChecker,
      "security.GoshawkChecker", "Detect DoubleFree, UAF by MOS.","",false);
    registry.addCheckerOption("string", "security.GoshawkChecker", "MemFuncsDir","/tmp/CSA","The directory that store the MOS.","alpha");
    registry.addCheckerOption("string", "security.GoshawkChecker", "PathNumberFile","/tmp/CSA/path_number","The path of file to save the number of paths that analyzed.","alpha");
    registry.addCheckerOption("string", "security.GoshawkChecker", "ExternFile","/tmp/CSA/extern_number","The number of MOS funcs that be modeled.","alpha");
    

}

extern "C" const char clang_analyzerAPIVersionString[] =
    CLANG_ANALYZER_API_VERSION_STRING;
