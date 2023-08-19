
#ifdef RCT_NEW_ARCH_ENABLED
#import "RNTfLiteSpec.h"

@interface TfLite : NSObject <NativeTfLiteSpec>
#else
#import <React/RCTBridgeModule.h>

@interface TfLite : NSObject <RCTBridgeModule>
#endif

@end
