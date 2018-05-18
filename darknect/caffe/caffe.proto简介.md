# caffe.proto这个文件
    它定义了caffe中用到的许多结构化数据。
    caffe采用了Protocol Buffers的数据格式。
    那么，Protocol Buffers到底是什么东西呢？简单说：
    Protocol Buffers 是一种轻便高效的结构化数据存储格式，可以用于结构化数据串行化，或者说序列化。

    简单地说，这个东东干的事儿其实和XML差不多，
    也就是把某种数据结构的信息，以某种格式保存起来。
    主要用于数据存储、传输协议格式等场合。
    有同学可能心理犯嘀咕了：
    放着好好的XML不用，干嘛重新发明轮子啊？！
    先别急，后面俺自然会有说道。
    
#  message  信息 假设订单包括如下属性：
    －－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－
      时间：time（用整数表示）
      客户id：userid（用整数表示）
      交易金额：price（用浮点数表示）
      交易的描述：desc（用字符串表示）
    －－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－
      如果使用protobuf实现，首先要写一个proto文件（不妨叫Order.proto），
      在该文件中添加一个名为"Order"的message结构，用来描述通讯协议中的结构化数据。
      该文件的内容大致如下：
    －－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－

    message Order
    {
      required int32 time = 1;
      required int32 userid = 2;
      required float price = 3;
      optional string desc = 4;
    }
    
# （一般来说，一个message结构会生成一个包装类）
      然后你使用类似下面的代码来序列化/解析该订单包装类：

    －－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－

    // 发送方

    Order order;
    order.set_time(XXXX);
    order.set_userid(123);
    order.set_price(100.0f);
    order.set_desc("a test order");

    string sOrder;
    order.SerailzeToString(&sOrder);

    // 然后调用某种socket的通讯库把序列化之后的字符串发送出去
    // ......

    －－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－

    // 接收方

    string sOrder;
    // 先通过网络通讯库接收到数据，存放到某字符串sOrder
    // ......

    Order order;
    if(order.ParseFromString(sOrder))  // 解析该字符串
    {
      cout << "userid:" << order.userid() << endl
              << "desc:" << order.desc() << endl;
    }
    else
    {
      cerr << "parse error!" << endl;
    }

    －－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－－


    有了这种代码生成机制，
    开发人员再也不用吭哧吭哧地编写那些协议解析的代码了（干这种活是典型的吃力不讨好）。

    万一将来需求发生变更，要求给订单再增加一个“状态”的属性，那只需要在Order.proto文件中增加一行代码。
    对于发送方（模块A），只要增加一行设置状态的代码；
    对于接收方（模块B）只要增加一行读取状态的代码。哇塞，简直太轻松了！
    另外，如果通讯双方使用不同的编程语言来实现，使用这种机制可以有效确保两边的模块对于协议的处理是一致的。
# 定义第一个Protocol Buffer消息。
    创建扩展名为.proto的文件，如：MyMessage.proto，并将以下内容存入该文件中。
    message LogonReqMessage {
      required int64 acctID = 1;
      required string passwd = 2;
    }
    这里将给出以上消息定义的关键性说明。
    1. message是消息定义的关键字，等同于C++中的struct/class，或是Java中的class。
    2. LogonReqMessage为消息的名字，等同于结构体名或类名。
    3. required前缀表示该字段为必要字段，既在序列化和反序列化之前该字段必须已经被赋值。
     与此同时，在Protocol Buffer中还存在另外两个类似的关键字，
     optional和repeated，带有这两种限定符的消息字段则没有required字段这样的限制。
    4. int64和string分别表示长整型和字符串型的消息字段， 在Protocol Buffer中存在一张类型对照表，
    既Protocol Buffer中的数据类型与其他编程语言(C++/Java)中所用类型的对照。
    该对照表中还将给出在不同的数据场景下，哪种类型更为高效。该对照表将在后面给出。
    5. acctID和passwd分别表示消息字段名，等同于Java中的域变量名，或是C++中的成员变量名。
    6. 标签数字1和2则表示不同的字段在序列化后的二进制数据中的布局位置。
    在该例中，passwd字段编码后的数据一定位于acctID之后。
    需要注意的是该值在同一message中不能重复。另外，对于Protocol Buffer而言，
    标签值为1到15的字段在编码时可以得到优化，既标签值和类型信息仅占有一个byte，
    标签范围是16到2047的将占有两个bytes，而Protocol Buffer可以支持的字段数量则为2的29次方减一。
    有鉴于此，我们在设计消息结构时，可以尽可能考虑让repeated类型的字段标签位于1到15之间，
    这样便可以有效的节省编码后的字节数量。

#  定义第二个（含有枚举字段 enum ）Protocol Buffer消息。
      //在定义Protocol Buffer的消息时，可以使用和C++/Java代码同样的方式添加注释。
      // enum是枚举类型定义的关键字，等同于C++/Java中的enum。
      // 和C++/Java中的枚举不同的是，枚举值之间的分隔符是分号，而不是逗号。
      enum UserStatus {// UserStatus为枚举的名字
          OFFLINE = 0;  //表示处于离线状态的用户
          ONLINE = 1;   //表示处于在线状态的用户
      }
      message UserInfo {
          required int64 acctID = 1;
          required string name = 2;
          required UserStatus status = 3;
      }
      
# 定义第三个（含有嵌套消息字段）Protocol Buffer消息。
     我们可以在同一个.proto文件中定义多个message，这样便可以很容易的实现嵌套消息的定义。
     如：
     
          enum UserStatus {
              OFFLINE = 0;
              ONLINE = 1;
          }
          message UserInfo {
              required int64 acctID = 1;
              required string name = 2;
              required UserStatus status = 3;
          }
          message LogonRespMessage {
              required LoginResult logonResult = 1;
              required UserInfo userInfo = 2;
          }
          
    这里将给出以上消息定义的关键性说明（仅包括上两小节中没有描述的）。
          1. LogonRespMessage消息的定义中包含另外一个消息类型作为其字段，如UserInfo userInfo。
          2. 上例中的UserInfo和LogonRespMessage被定义在同一个.proto文件中，
             那么我们是否可以包含在其他.proto文件中定义的message呢？
             Protocol Buffer提供了另外一个关键字import，
             这样我们便可以将很多通用的message定义在同一个.proto文件中，
             而其他消息定义文件可以通过import的方式将该文件中定义的消息包含进来
             ，如：
          import "myproject/CommonMessages.proto"    
# 限定符(required/optional/repeated)的基本规则。
      1. 在每个消息中必须至少留有一个required类型的字段。 
      2. 每个消息中可以包含0个或多个optional类型的字段。
      3. repeated表示的字段可以包含0个或多个数据。
         需要说明的是，这一点有别于C++/Java中的数组，因为后两者中的数组必须包含至少一个元素。
      4. 如果打算在原有消息协议中添加新的字段，同时还要保证老版本的程序能够正常读取或写入，
         那么对于新添加的字段必须是optional或repeated。
         道理非常简单，老版本程序无法读取或写入新增的required限定符的字段。
         
# 类型对照表。
    .protoType	Notes	C++ Type	Java Type
    double	 	          double	 double
    float	 	          float	     float
    int32	          	  int32	     int
    int64	          	  int64	     long
    uint32	              uint32	 int
    uint64	              uint64	 long
    sint32	              int32	     int
    sint64	         	  int64	     long
    fixed32	  > 228. 	  uint32	 int
    fixed64	 > 256.	      uint64	 long
    sfixed32	4字节  	int32	   int
    sfixed64	8字节 	int64	   long
    bool	 	          bool	     boolean
    string	 UTF-8  ASCII  string	 String
    bytes	         	string	 ByteString

# Protocol Buffer消息升级原则。
      在实际的开发中会存在这样一种应用场景，既消息格式因为某些需求的变化而不得不进行必要的升级，
      但是有些使用原有消息格式的应用程序暂时又不能被立刻升级，
      这便要求我们在升级消息格式时要遵守一定的规则，从而可以保证基于新老消息格式的新老程序同时运行。
      规则如下：
      1. 不要修改已经存在字段的标签号。
      2. 任何新添加的字段必须是optional和repeated限定符，
         否则无法保证新老程序在互相传递消息时的消息兼容性。
      3. 在原有的消息中，不能移除已经存在的required字段，
         optional和repeated类型的字段可以被移除，但是他们之前使用的标签号必须被保留，不能被新的字段重用。
      4. int32、uint32、int64、uint64和bool等类型之间是兼容的，
         sint32和sint64是兼容的，string和bytes是兼容的，fixed32和sfixed32，
         以及fixed64和sfixed64之间是兼容的，这意味着如果想修改原有字段的类型时，
         为了保证兼容性，只能将其修改为与其原有类型兼容的类型，否则就将打破新老消息格式的兼容性。
      5. optional和repeated限定符也是相互兼容的。
      
# 包 命名空间 Packages。
      我们可以在.proto文件中定义包名，如：
      package ourproject.lyphone;
      该包名在生成对应的C++文件时，将被替换为名字空间名称，
      既namespace ourproject { namespace lyphone。而在生成的Java代码文件中将成为包名。
      
# Options。
    Protocol Buffer允许我们在.proto文件中定义一些常用的选项，
    这样可以指示Protocol Buffer编译器帮助我们生成更为匹配的目标语言代码。

    Protocol Buffer内置的选项被分为以下三个级别：

    1. 文件级别，这样的选项将影响当前文件中定义的所有消息和枚举。
    2. 消息级别，这样的选项仅影响某个消息及其包含的所有字段。
    3. 字段级别，这样的选项仅仅响应与其相关的字段。

    3. option optimize_for = LITE_RUNTIME;
    optimize_for是文件级别的选项，
    Protocol Buffer定义三种优化级别SPEED/CODE_SIZE/LITE_RUNTIME。
    缺省情况下是SPEED。

      SPEED: 表示生成的代码运行效率高，但是由此生成的代码编译后会占用更多的空间。
      CODE_SIZE: 和SPEED恰恰相反，代码运行效率较低，但是由此生成的代码编译后会占用更少的空间，
                 通常用于资源有限的平台，如Mobile。
      LITE_RUNTIME: 生成的代码执行效率高，同时生成代码编译后的所占用的空间也是非常少。
                    这是以牺牲Protocol Buffer提供的反射功能为代价的。
                    因此我们在C++中链接Protocol Buffer库时仅需链接libprotobuf-lite，而非libprotobuf。
                    在Java中仅需包含protobuf-java-2.4.1-lite.jar，而非protobuf-java-2.4.1.jar。
    注：对于LITE_MESSAGE选项而言，其生成的代码均将继承自MessageLite，而非Message。 

    4. [pack = true]: 因为历史原因，对于数值型的repeated字段，如int32、int64等，
                    在编码时并没有得到很好的优化，然而在新近版本的Protocol Buffer中，
                    可通过添加[pack=true]的字段选项，
                    以通知Protocol Buffer在为该类型的消息对象编码时更加高效。
                    如：
                    repeated int32 samples = 4 [packed=true]。
             注：该选项仅适用于2.3.0以上的Protocol Buffer。
    5. [default = default_value]: optional类型的字段，如果在序列化时没有被设置，
                    或者是老版本的消息中根本不存在该字段，那么在反序列化该类型的消息是，
                    optional的字段将被赋予类型相关的缺省值，如bool被设置为false，
                    int32被设置为0。
                    Protocol Buffer也支持自定义的缺省值，
                    如：
             optional int32 result_per_page = 3 [default = 10]。

# 命令行编译工具。
      protoc --proto_path=IMPORT_PATH --cpp_out=DST_DIR --java_out=DST_DIR --python_out=DST_DIR path/to/file.proto
      这里将给出上述命令的参数解释。
      1. protoc       为Protocol Buffer提供的命令行编译工具。
      2. --proto_path 等同于-I选项，主要用于指定待编译的.proto消息定义文件所在的目录，该选项可以被同时指定多个。
      3. --cpp_out    选项表示生成C++代码，
                     --java_out表示生成Java代码，
                     --python_out则表示生成Python代码，
                     其后的目录为生成后的代码所存放的目录。
                     
      4. path/to/file.proto表示待编译的消息定义文件。
      
      注：对于C++而言，通过Protocol Buffer编译工具，
      可以将每个.proto文件生成出一对.h和.cc的C++代码文件。
      生成后的文件可以直接加载到应用程序所在的工程项目中。
      如：MyMessage.proto 生成的文件为 MyMessage.pb.h 和 MyMessage.pb.cc。

[Caffe代码解析](http://alanse7en.github.io/caffedai-ma-jie-xi-2/)

 # caffe.proto 示例
    syntax = "proto2";
    // 第一行proto3表示用的是proto3，默认是proto2
    package caffe;

    // Specifies the shape (dimensions) of a Blob.
    message BlobShape {
      repeated int64 dim = 1 [packed = true];
    }
