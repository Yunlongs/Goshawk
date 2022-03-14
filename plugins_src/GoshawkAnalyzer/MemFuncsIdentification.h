//#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <iostream>
#include "json.hpp"
using json = nlohmann::json;
#include <tsl/robin_map.h>
#include <sys/fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <sstream>

//using namespace clang;
//using namespace ento;
//using namespace loc;

#define debug false

enum FuncKind{
    Normal,Customized
};


struct MallocEntry{
  FuncKind kind;
  std::vector<std::string> MemberVector;
  std::vector<std::pair<int,std::string>> ParameterVector;
  std::vector<std::pair<int,std::string>> ParamMemberVector;
};

struct FreeEntry{
  FuncKind kind;
  std::vector<int> ParamIndex;
  std::vector<std::string> ParamMeberVector;
};

/*
MemFuncsUtility is used to check whether a function is memory allocate function
or a memory free function since most librarys implement their own memory functions
*/
class MemFuncsUtility
{
  private:
  std::string Memfunc_dir;         // Directory which stores memory files.
  std::string AllocNormalFile;     // Alloc memory only to return value. Like: malloc();
  std::string AllocCustomizedFile; // Alloc memory to Sturcture member or Parameters.
  std::string FreeNormalFile;      // Release memory only with it's first parameters. Like: free();
  std::string FreeCustomizedFile;  // Release memory with parameters or structure members.
  std::string ReallocFile;


  public:
  //std::map<std::string,struct MallocEntry> AllocFuncs;
  //std::map<std::string,struct FreeEntry> FreeFuncs;
  tsl::robin_map<std::string,struct MallocEntry,std::hash<std::string>,std::equal_to<std::string>,
      std::allocator<std::pair<std::string, struct MallocEntry>>,true>  AllocFuncs;
  tsl::robin_map<std::string,struct FreeEntry,std::hash<std::string>,std::equal_to<std::string>,
     std::allocator<std::pair<std::string, struct FreeEntry>>,true>  FreeFuncs;
  std::vector<std::string> ReallocFuncs;
  
  MemFuncsUtility(std::string Dir){Memfunc_dir = Dir;InitFiles();InitFuncs();}
  void InitFuncs();
  void InitFiles();
  //std::string getFunctionNameFromCall(const CallEvent &event);
  struct FreeEntry* isFreeFunction(std::string func_name);
  //int isFreeFunction(const CallEvent &event);
  bool isReallocFunction(std::string func_name);
  //bool isReallocFunction(const CallEvent &event);
  struct MallocEntry* isMallocFunction(std::string func_name);
  //int isMallocFunction(const CallEvent &event);
};

void strip(std::string &str)
{
  if (str[0] == '-')
    str.erase(0,2);
  else
    str.erase(0,1);
}

void MemFuncsUtility::InitFiles()
{
  AllocNormalFile = Memfunc_dir + "/AllocNormalFile.txt";
  AllocCustomizedFile = Memfunc_dir + "/AllocCustomizedFile.txt";
  FreeNormalFile = Memfunc_dir + "/FreeNormalFile.txt";
  FreeCustomizedFile = Memfunc_dir + "/FreeCustomizedFile.txt";
  ReallocFile = Memfunc_dir + "/ReallocFile.txt";
}

void MemFuncsUtility::InitFuncs()
{
  std::ifstream file;

  /*
    For Normal malloc functions,
    each line represent a memory allocate function name
  */
  file.open(AllocNormalFile);
  if(!file) {std::cout<<"Can't open mem allocation file!. Please check this path:"<<AllocNormalFile<<"\n";}
  std::string readline;
  while (getline(file, readline))  
  {
    std::string funcname = readline;
    struct MallocEntry entry;
    entry.kind = Normal;
    AllocFuncs.insert(std::pair<std::string, struct MallocEntry>(funcname,entry));
  }
  file.close();

  /*
    For Customized alloc functions,
    each line repersents a alloc json string.
  */
  file.open(AllocCustomizedFile);
  if(!file) {std::cout<<"Can't open customized mem allocation file!. Please check this path:"<<AllocCustomizedFile<<"\n";}
  while (getline(file, readline))
  {
    auto func_json = json::parse(readline);
    std::string funcname = func_json["funcname"];
    struct MallocEntry entry;
    entry.kind = Customized;

    auto returned_objects = func_json["returned_object"];
    for (int i=0; i < returned_objects.size(); i++)
    {
      std::string member_name = returned_objects[i];
      strip(member_name);
      entry.MemberVector.push_back(member_name);
    }

    auto param_objects = func_json["param_object"];
    for (int i=0; i< param_objects.size(); i+=2)
    {
      int arg_index =param_objects[i];
      std::string name = param_objects[i+1];
      if (name[0] =='-' or name[0] == '.') // is a member name.
      {
        strip(name);
        entry.ParamMemberVector.push_back(std::pair<int, std::string>(arg_index -1, name));
      }
      else // is a parameter's name
      {
        entry.ParameterVector.push_back(std::pair<int,std::string>(arg_index,name));
      }
    }
    AllocFuncs.insert(std::pair<std::string, struct MallocEntry>(funcname,entry));
  }
  file.close();
  if (debug) {std::cout<<"Allocation Count: "<<AllocFuncs.size()<<"\n";}


  /*
    For Normal Free functions, each line is a function's name.
  */
  file.open(FreeNormalFile);
  if(!file){std::cout<<"Can't open mem deallocation file! Please check this path: "<<FreeNormalFile<<"\n";}
  while (getline(file, readline)) 
  {
    std::string funcname = readline;
    FreeEntry entry;
    entry.kind = Normal;
    FreeFuncs.insert(std::pair<std::string, struct FreeEntry>(funcname,entry));
  }
  file.close();

  /*
    For Customized Free functions, each line is a json string.
  */
  file.open(FreeCustomizedFile);
  if(!file){std::cout<<"Can't open mem FreeCustomizedFile file! Please check this path: "<<FreeCustomizedFile<<"\n";}
  while (getline(file, readline))
  {
    auto func = json::parse(readline);
    std::string funcname = func["funcname"];
    auto param_names = func["param_names"];
    auto member_name_list = func["member_name"];
    struct FreeEntry entry;
    entry.kind = Customized;
    for (int i=0; i<param_names.size(); i+=2)
    {
      int index = param_names[i];
      entry.ParamIndex.push_back(index);
    }

    //if (debug){std::cout<<"param_name:\t"<<param_name<<"\n";}
    //if (debug){std::cout<<"member_name_list:\t"<<member_name_list.dump()<<"\n";}
    for (std::string member_name: member_name_list)
    {
      //if(debug){std::cout<<"current member name:\t"<<member_name<<"\n";}
      entry.ParamMeberVector.push_back(member_name);
    }
    FreeFuncs.insert(std::pair<std::string, struct FreeEntry>(funcname,entry));
  }
  file.close();

  if (debug) {std::cout<<"Deallocation Count: "<<FreeFuncs.size()<<"\n";}

  //file.open(ReallocFile);
  //if(!file){std::cout<<"Can't open mem reallocation file! Please check this path: "<<ReallocFile<<"\n";}
  //while (getline(file, readline)) 
  //{
   // ReallocFuncs.push_back(readline);
  //}
  //file.close();
  std::string str = "realloc";
  ReallocFuncs.push_back(str);
  if (debug) {std::cout<<"Reallocation Count: "<<ReallocFuncs.size()<<"\n";}
}


/*
  This function does not support get the indirect call's name. Such as func->release().
*//*
std::string MemFuncsUtility::getFunctionNameFromCall(const CallEvent &event)
{
  const Decl * decl = event.getDecl();
  if(!decl)
    return "";
  const FunctionDecl *func_decl = decl->getAsFunction();
  if(!func_decl)
    return "";
  std::string func_name = func_decl->getQualifiedNameAsString();
  return func_name;
}
*/


/*
  Check if the function is in the Alloc Functions set.
  If not, return 0. Normal kind return 1, Customized kind return 2.
*/
/*
int MemFuncsUtility::isMallocFunction(const CallEvent &event)
{
  std::string func_name = getFunctionNameFromCall(event);
  return isMallocFunction(func_name);
}*/

struct MallocEntry* MemFuncsUtility::isMallocFunction(std::string func_name)
{
  auto iter = AllocFuncs.find(func_name);
  if (iter != AllocFuncs.end()) // Not in AllocFuncs.
  {
    const MallocEntry *entry = &(iter->second);
    return const_cast<MallocEntry*>(entry);
  }
  return NULL;
}




/*
  Check if the function is in the Free Functions set.
  If not, return 0. Normal kind return 1, Customized kind return 2.
*/
struct FreeEntry* MemFuncsUtility::isFreeFunction(std::string func_name)
{
  auto iter = FreeFuncs.find(func_name);
  if (iter != FreeFuncs.end()) // Not in FreeFuncs.
  {
    const FreeEntry* entry = &(iter->second);
    return const_cast<FreeEntry*>(entry);
  }
  return NULL;
}
/*
int MemFuncsUtility::isFreeFunction(const CallEvent &event)
{
  std::string func_name = getFunctionNameFromCall(event);
  return isFreeFunction(func_name);
}*/

bool MemFuncsUtility::isReallocFunction(std::string func_name)
{
  for (std::string realloc_name : ReallocFuncs)
  {
    if (realloc_name == func_name)
      return true;
  }
  return false;
}
/*
bool MemFuncsUtility::isReallocFunction(const CallEvent &event)
{
  
    std::string func_name = getFunctionNameFromCall(event);
    return isReallocFunction(func_name);
}*/

void write_number_line(long long int num, std::string file)
{
  std::ofstream out_file;
  out_file.open(file,std::ios::app);
  if(!out_file){llvm::errs()<<"Can't open file:"<<file<<"\n";}
  std::stringstream ss;
  ss<<num<<"\n";
  out_file<<ss.str();
  out_file.close();
}


void write_number(long long int num, const char* file)
{
   // get the file handle
   int fd = open(file, O_RDWR);
	if (fd == -1) {
		perror("open");
		return ;
	}


    // 锁住整个文件
	struct flock lock = {};
    lock.l_type = F_WRLCK;
    lock.l_whence = SEEK_SET;
    lock.l_start = 0;
    lock.l_len = 0;

    lock.l_pid = getpid();
    
    //lock the file.
    if (fcntl(fd, F_SETLKW, &lock) == -1) {
        perror("fcntl");
        return ;
    }

    long long int count;
    std::stringstream ss;
    char buf[100];

    //read the num
    read(fd,buf,100);
    std::string string_count = buf;
    ss<<string_count;
    ss>>count;
    ss.clear();
    
    //write the num;
    //std::cout<<"read count:"<<count<<std::endl;
    count += num;
    ss<<count;
    ss>>string_count;
    ss.clear();
    const char* new_buf = string_count.data();
    lseek(fd,0,SEEK_SET);
    write(fd,new_buf,100);
    //std::cout<<"write count:"<<count<<std::endl;
   // 释放锁
    lock.l_type = F_UNLCK;
    if (fcntl(fd, F_SETLKW, &lock) == -1) {
        perror("fcntl");
        return ;
    }
    close(fd);
}