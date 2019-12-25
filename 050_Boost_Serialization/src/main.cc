#include <iostream>

#include <boost/serialization/vector.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
 
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <string>
#include <iostream>
#include <fstream>
 
using namespace std;
 
class CSerialize
{
 
private:
	
	friend class boost::serialization::access;
 
	//数据成员
	int m_nNum1;
	double m_dNum2;
	std::string m_strInfo;
 
	template<class Archive>
	void serialize( Archive & ar, const unsigned int file_version )
	{
		ar & m_nNum1;
		ar & m_dNum2;
		ar & m_strInfo;
	}
public:
	CSerialize(int _num1, double _num2, string& _strInfo) : m_nNum1(_num1), m_dNum2(_num2), m_strInfo(_strInfo){
		
	}
	CSerialize(){}
	~CSerialize(){};
 
	void print()
	{
		cout << m_nNum1 << " " << m_dNum2 << endl;
		cout << m_strInfo << endl;
	}
 
	static void saveData(CSerialize& data, string fileName)
	{
		ofstream fout(fileName);
		boost::archive::binary_oarchive ar_out(fout);
		ar_out << data;
	}
	static void loadData(CSerialize& data, string fileName)
	{
		ifstream fin(fileName);
		boost::archive::binary_iarchive ar_in(fin);
		ar_in >> data;
	}
 
 
	ofstream& operator<<(ofstream& os);
	ifstream& operator>>(ifstream& is);
protected:
private:
};
 
ofstream& CSerialize::operator<<(ofstream& os)
{
	os << m_nNum1 << m_dNum2 << m_strInfo;
	return os;
}
ifstream& CSerialize::operator>>(ifstream& is)
{
	is >> m_nNum1 >> m_dNum2 >> m_strInfo;
	return is;
}
int main(int argc, char *argv[])
{
    cout<<"Boost Serialization test."<<endl;
    cout<<"Complied at "<<__TIME__<<", "<<__DATE__<<"."<<endl;

	ofstream fout("test.vipp");
	boost::archive::binary_oarchive ar_out(fout);
	
    std::string str("Oh Oh ha ha tttt");
	CSerialize obj1(1, 2.5, str);
	ar_out << obj1;
	fout.close();
 
 
	ifstream fin("test.vipp");
	boost::archive::binary_iarchive ar_in(fin);
	CSerialize new_obj;
	ar_in >> new_obj;
	fin.close();
	new_obj.print();
 
	CSerialize test_obj;
	CSerialize::loadData(test_obj, "test.vipp");
	test_obj.print();
	getchar();
	return 0;
}