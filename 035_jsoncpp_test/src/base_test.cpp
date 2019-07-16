#include <json/json.h>
#include <iostream>
#include <fstream>

using namespace std;
// using namespace Json;

int main(int argc, char* argv[])
{
    cout<<"Read and Parse Json file."<<endl;
    cout<<"Compiled at "<<__TIME__<<", "<<__DATE__<<"."<<endl;

    if(argc!=2)
    {
        cout<<"Usage: "<<argv[0]<<" json_file"<<endl;
        return 0;
    }





    return 0;
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




