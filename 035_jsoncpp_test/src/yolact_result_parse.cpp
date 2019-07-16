#include <json/json.h>
#include <json/value.h>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

int Json_ReadInt(Json::Value JV, int ori_value = 0);
double Json_ReadDouble(Json::Value JV, double ori_value = 0.0);
string Json_ReadString(Json::Value JV, string ori_value = "");
bool Json_ReadBool(Json::Value JV, bool ori_value = true);

int main(int argc, char* argv[])
{
    cout<<"Read and Parse Json file."<<endl;
    cout<<"Compiled at "<<__TIME__<<", "<<__DATE__<<"."<<endl;

    if(argc!=2)
    {
        cout<<"Usage: "<<argv[0]<<" json_file"<<endl;
        return 0;
    }

	ifstream fin;
	fin.open(argv[1]);
	if(!fin)
	{
		cout<<"[Fatal] Json file: "<<argv[1]<<" open failed."<<endl;
	}
	else
	{
		cout<<"[Info ] Json file: "<<argv[1]<<" open ... ok."<<endl;
	}

	Json::Value JsonRoot,JsonObjLoop;
	// 建议使用这样子的解析方式
	try
	{
		// 如果发生错误，直接在内部就。。。
		fin >> JsonRoot;
		fin.close();
	}
	catch( ... )
	{
		cout<<"Error occured when parsing json file."<<endl;
		// cout<<"\twhat(): "<< e.what()<<endl;
		return 0;
	}

    // 准备读取数据，获取Json文件中的数据总量
    int i=-1;
    stringstream ss;

    do
    {
        ++i;
        ss.clear();
        ss.str("");
        ss<<i;
    } while (JsonRoot[ss.str()].isObject());

    cout<<"[info ] Total json object: "<<i<<endl;

    // 这里现在只是在读取第0个物体的数据
	Json::Value JsonObj1 = JsonRoot["0"];
    cout<<"category_id: "<<JsonObj1["category_id"]<<endl;
    Json::Value JsonBBoxArr = JsonObj1["bbox"];
    cout<<"bbox: ("<<JsonBBoxArr[0]<<", "<<JsonBBoxArr[1]<<"), ("<<JsonBBoxArr[2]<<", "<<JsonBBoxArr[3]<<")"<<endl;
    cout<<"segmentation.size=("<<JsonObj1["segmentation"]["size"][0]<<", "<<JsonObj1["segmentation"]["size"][1]<<")"<<endl;
    

    return 0;

    


	// 普通对象
	cout << "encoding = " << Json_ReadString(JsonRoot["encoding"]) << endl;

	// 元组对象
	Json::Value JsonIndent = JsonRoot["indent"];
	cout << "indent.length = " << Json_ReadInt(JsonIndent["length"]) << endl;
	cout << "indent.use_space = " << (Json_ReadBool(JsonIndent["use_space"])?"True":"False") << endl;

	Json::Value ArrString = JsonRoot["plug-ins"];
	cout<<"plug-ins:"<<endl;
	// 注意要使用Json::ArrayIndex
	for(Json::ArrayIndex i=0;i<ArrString.size();++i)
	{
		cout<<"\t"<<Json_ReadString(ArrString[i])<<endl;
	}

	// 尝试读取不存在的：
	// Json::Value JsonItem1 = JsonRoot["1"];

	if(JsonRoot["indent233"].isObject())
	{
		cout<<"indent233 ok."<<endl;
	}
	else
	{
		cout<<"indent233 failed."<<endl;
	}
	

	// cout << "1.name = " << Json_ReadString(JsonItem1["name"]) << endl;
	// cout << "1.name2 = " << Json_ReadString(JsonItem1["name2"]) << endl;

	// 可以使用这种方式来访问某个项
	cout << "indent.length = " << JsonRoot["indent"]["length"] << endl;

    return 0;
}

int Json_ReadInt(Json::Value JV, int ori_value)
{
	int result = ori_value;
	Json::ValueType VT = JV.type();
	if (VT == Json::ValueType::intValue)
		result = JV.asInt();
	return result;
}
double Json_ReadDouble(Json::Value JV, double ori_value)
{
	double result = ori_value;
	Json::ValueType VT = JV.type();
	if (VT == Json::ValueType::realValue)
		result = JV.asDouble();
	return result;
}
string Json_ReadString(Json::Value JV, string ori_value)
{
	string result = ori_value;
	Json::ValueType VT = JV.type();
	if (VT == Json::ValueType::stringValue)
		result = JV.asCString();
	return result;
}
bool Json_ReadBool(Json::Value JV, bool ori_value)
{
	bool result = ori_value;
	Json::ValueType VT = JV.type();
	if (VT == Json::ValueType::booleanValue)
		result = JV.asBool();
	return result;
}


/*

参考代码

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include "json\json.h"
using namespace std;


int Json_ReadInt(Json::Value JV, int ori_value = 0);
double Json_ReadDouble(Json::Value JV, double ori_value = 0.0);
string Json_ReadString(Json::Value JV, string ori_value = "");
bool Json_ReadBool(Json::Value JV, bool ori_value = true);
// string UnicodeToANSI(CString strUnicode);
// CString ANSIToUnicode(string strANSI);


void ReadJsonFile()
{
	ifstream fin;
	fin.open("jsonfile.json");
	if (!fin)
	{
		//TCHAR("error");
		return;
	}
	ostringstream ostring;
	ostring << fin.rdbuf();
	fin.close();
	string strContext = ostring.str();
	// CharReaderBuilder
	Json::CharReaderBuilder builder;
	Json::CharReader* JsonReader(builder.newCharReader());
	Json::Value JsonRoot, ObjectTmp;
	JSONCPP_STRING errs;
	const char* pstr = strContext.c_str();
	if (!JsonReader->parse(pstr, pstr + strlen(pstr), &JsonRoot, &errs))
	{
		//TCHAR("error");
		return;
	}
	//..//
	string stringTmp; // ��ӡ�ַ���
	int intTmp; // ��ӡ����
	double doubleTmp; // ��ӡ������
	bool boolTmp; // ��ӡ������
	// ��ȡ�ַ���
	stringTmp = Json_ReadString(JsonRoot["name"]);
	cout << "name = " << stringTmp << endl;
	// ��ȡ����
	intTmp = Json_ReadInt(JsonRoot["age"]);
	cout << "age = " << intTmp << endl;
	// ��ȡ����������
	doubleTmp = Json_ReadDouble(JsonRoot["height"]);
	cout << "height = " << doubleTmp << endl;
	// ��ȡ������
	boolTmp = Json_ReadBool(JsonRoot["play_football"]);
	cout << "play_football = " << boolTmp << endl;
	// ��ȡJson����
	Json::Value JsonObj = JsonRoot["object"];
	intTmp = Json_ReadInt(JsonObj["sonetime"]);
	stringTmp = Json_ReadString(JsonObj["someone"]);
	stringTmp = Json_ReadString(JsonObj["somewhere"]);
	// ��ȡ�������飬�ȶ�ȡ�������Ȼ���ڶ����ڱ���
	Json::Value ArrInt = JsonRoot["number_array"];
	for (size_t i = 0; i < ArrInt.size(); i++)
	{
		intTmp = Json_ReadInt(ArrInt[i]);
	}
	// ��ȡ�ַ�������
	Json::Value ArrString = JsonRoot["string_array"];
	for (size_t j = 0; j < ArrString.size(); j++)
	{
		stringTmp = Json_ReadString(ArrString[j]);
	}
	// ��ȡJson��������
	Json::Value ObjectArray;
	ObjectArray = JsonRoot["object_array"];
	for (size_t k = 0; k < ObjectArray.size(); k++)
	{
		stringTmp = Json_ReadString(ObjectArray[k]["string1"]);
		stringTmp = Json_ReadString(ObjectArray[k]["string2"]);
	}
}

void WriteJsonFile()
{
	// ����Json�����������Ϊ��
	Json::Value JsonRoot;
	// д���ַ���
	JsonRoot["name"] = Json::Value("Denny");
	// д������
	JsonRoot["age"] = Json::Value(22);
	// д�븡��������
	JsonRoot["height"] = Json::Value(1.78);
	// д�벼����
	JsonRoot["play_football"] = Json::Value(true);
	// д��Json����
	Json::Value JsonObj;
	JsonObj["sometime"] = Json::Value(2018);
	JsonObj["someone"] = Json::Value("Kelly");
	JsonObj["somewhere"] = Json::Value("city");
	JsonRoot["object"] = JsonObj;
	// ������д����������
	JsonRoot["number_array"].append(1);
	JsonRoot["number_array"].append(2);
	JsonRoot["number_array"].append(3);
	JsonRoot["number_array"].append(4);
	// ������д���ַ�������
	JsonRoot["string_array"].append("string01");
	JsonRoot["string_array"].append("string02");
	JsonRoot["string_array"].append("string03");
	// д��Json�������飬�������ɶ��󹹳�
	Json::Value JsonArr1, JsonArr2, JsonArr3;
	JsonArr1["string1"] = Json::Value("1-1");
	JsonArr1["string2"] = Json::Value("1-2");
	JsonArr2["string1"] = Json::Value("2-1");
	JsonArr2["string2"] = Json::Value("2-2");
	JsonArr3["string1"] = Json::Value("3-1");
	JsonArr3["string2"] = Json::Value("3-2");
	JsonRoot["object_array"].append(JsonArr1);
	JsonRoot["object_array"].append(JsonArr2);
	JsonRoot["object_array"].append(JsonArr3);
	// ����Json�ļ�����
	ofstream fout("jsonfile.json");
	if (fout)
	{
		string strContext;
		strContext = JsonRoot.toStyledString();
		fout << strContext;
		fout.close();
	}
}

int main()
{
	WriteJsonFile();
	ReadJsonFile();
	return 0;
}

///////////////////////////////////////////////////
int Json_ReadInt(Json::Value JV, int ori_value)
{
	int result = ori_value;
	Json::ValueType VT = JV.type();
	if (VT == Json::ValueType::intValue)
		result = JV.asInt();
	return result;
}
double Json_ReadDouble(Json::Value JV, double ori_value)
{
	double result = ori_value;
	Json::ValueType VT = JV.type();
	if (VT == Json::ValueType::realValue)
		result = JV.asDouble();
	return result;
}
string Json_ReadString(Json::Value JV, string ori_value)
{
	string result = ori_value;
	Json::ValueType VT = JV.type();
	if (VT == Json::ValueType::stringValue)
		result = JV.asCString();
	return result;
}
bool Json_ReadBool(Json::Value JV, bool ori_value)
{
	bool result = ori_value;
	Json::ValueType VT = JV.type();
	if (VT == Json::ValueType::booleanValue)
		result = JV.asBool();
	return result;
}

// VS2015
// string UnicodeToANSI(CString strUnicode)
// {
// 	USES_CONVERSION;
// 	std::string strANSI;
// 	strANSI = W2A(strUnicode.GetString());
// 	return strANSI;
// }

// CString ANSIToUnicode(string strANSI)
// {
// 	CString strUnicode;
// 	strUnicode = strANSI.c_str();
// 	return strUnicode;
// }
  
*/




