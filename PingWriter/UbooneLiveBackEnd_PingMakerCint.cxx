// Do NOT change. Changes will be lost next time file is generated

#define R__DICTIONARY_FILENAME UbooneLiveBackEnd_PingMakerCint

/*******************************************************************/
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#define G__DICTIONARY
#include "RConfig.h"
#include "TClass.h"
#include "TDictAttributeMap.h"
#include "TInterpreter.h"
#include "TROOT.h"
#include "TBuffer.h"
#include "TMemberInspector.h"
#include "TInterpreter.h"
#include "TVirtualMutex.h"
#include "TError.h"

#ifndef G__ROOT
#define G__ROOT
#endif

#include "RtypesImp.h"
#include "TIsAProxy.h"
#include "TFileMergeInfo.h"
#include <algorithm>
#include "TCollectionProxyInfo.h"
/*******************************************************************/

#include "TDataMember.h"

// Since CINT ignores the std namespace, we need to do so in this file.
namespace std {} using namespace std;

// Header files passed as explicit arguments
#include "pngwriter.h"

// Header files passed via #pragma extra_include

namespace ROOT {
   static TClass *pngwriter_Dictionary();
   static void pngwriter_TClassManip(TClass*);
   static void *new_pngwriter(void *p = 0);
   static void *newArray_pngwriter(Long_t size, void *p);
   static void delete_pngwriter(void *p);
   static void deleteArray_pngwriter(void *p);
   static void destruct_pngwriter(void *p);

   // Function generating the singleton type initializer
   static TGenericClassInfo *GenerateInitInstanceLocal(const ::pngwriter*)
   {
      ::pngwriter *ptr = 0;
      static ::TVirtualIsAProxy* isa_proxy = new ::TIsAProxy(typeid(::pngwriter));
      static ::ROOT::TGenericClassInfo 
         instance("pngwriter", "pngwriter.h", 84,
                  typeid(::pngwriter), DefineBehavior(ptr, ptr),
                  &pngwriter_Dictionary, isa_proxy, 4,
                  sizeof(::pngwriter) );
      instance.SetNew(&new_pngwriter);
      instance.SetNewArray(&newArray_pngwriter);
      instance.SetDelete(&delete_pngwriter);
      instance.SetDeleteArray(&deleteArray_pngwriter);
      instance.SetDestructor(&destruct_pngwriter);
      return &instance;
   }
   TGenericClassInfo *GenerateInitInstance(const ::pngwriter*)
   {
      return GenerateInitInstanceLocal((::pngwriter*)0);
   }
   // Static variable to force the class initialization
   static ::ROOT::TGenericClassInfo *_R__UNIQUE_(Init) = GenerateInitInstanceLocal((const ::pngwriter*)0x0); R__UseDummy(_R__UNIQUE_(Init));

   // Dictionary for non-ClassDef classes
   static TClass *pngwriter_Dictionary() {
      TClass* theClass =::ROOT::GenerateInitInstanceLocal((const ::pngwriter*)0x0)->GetClass();
      pngwriter_TClassManip(theClass);
   return theClass;
   }

   static void pngwriter_TClassManip(TClass* ){
   }

} // end of namespace ROOT

namespace ROOT {
   // Wrappers around operator new
   static void *new_pngwriter(void *p) {
      return  p ? new(p) ::pngwriter : new ::pngwriter;
   }
   static void *newArray_pngwriter(Long_t nElements, void *p) {
      return p ? new(p) ::pngwriter[nElements] : new ::pngwriter[nElements];
   }
   // Wrapper around operator delete
   static void delete_pngwriter(void *p) {
      delete ((::pngwriter*)p);
   }
   static void deleteArray_pngwriter(void *p) {
      delete [] ((::pngwriter*)p);
   }
   static void destruct_pngwriter(void *p) {
      typedef ::pngwriter current_t;
      ((current_t*)p)->~current_t();
   }
} // end of namespace ROOT for class ::pngwriter

namespace {
  void TriggerDictionaryInitialization_libUbooneLiveBackEnd_PingMaker_Impl() {
    static const char* headers[] = {
"pngwriter.h",
0
    };
    static const char* includePaths[] = {
"/home/cadams/larlite/core",
"/home/cadams/Software/ROOT/root6/include/root",
"/home/cadams/larlite/UserDev/UbooneLiveBackEnd/PingWriter/",
0
    };
    static const char* fwdDeclCode = 
R"DICTFWDDCLS(
#pragma clang diagnostic ignored "-Wkeyword-compat"
#pragma clang diagnostic ignored "-Wignored-attributes"
#pragma clang diagnostic ignored "-Wreturn-type-c-linkage"
extern int __Cling_Autoloading_Map;
class __attribute__((annotate("$clingAutoload$pngwriter.h")))  pngwriter;
)DICTFWDDCLS";
    static const char* payloadCode = R"DICTPAYLOAD(

#ifndef G__VECTOR_HAS_CLASS_ITERATOR
  #define G__VECTOR_HAS_CLASS_ITERATOR 1
#endif
#ifndef NO_FREETYPE
  #define NO_FREETYPE 1
#endif

#define _BACKWARD_BACKWARD_WARNING_H
#include "pngwriter.h"

#undef  _BACKWARD_BACKWARD_WARNING_H
)DICTPAYLOAD";
    static const char* classesHeaders[]={
"pngwriter", payloadCode, "@",
nullptr};

    static bool isInitialized = false;
    if (!isInitialized) {
      TROOT::RegisterModule("libUbooneLiveBackEnd_PingMaker",
        headers, includePaths, payloadCode, fwdDeclCode,
        TriggerDictionaryInitialization_libUbooneLiveBackEnd_PingMaker_Impl, {}, classesHeaders);
      isInitialized = true;
    }
  }
  static struct DictInit {
    DictInit() {
      TriggerDictionaryInitialization_libUbooneLiveBackEnd_PingMaker_Impl();
    }
  } __TheDictionaryInitializer;
}
void TriggerDictionaryInitialization_libUbooneLiveBackEnd_PingMaker() {
  TriggerDictionaryInitialization_libUbooneLiveBackEnd_PingMaker_Impl();
}
