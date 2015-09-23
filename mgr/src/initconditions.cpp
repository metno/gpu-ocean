#include "initconditions.h"

using namespace std;

struct InitConditions::InitConditionsImpl
{
    FieldPtr field;
    InitConditionsImpl();
};

InitConditions::InitConditionsImpl::InitConditionsImpl()
    : field(new vector<float>)
{
};

InitConditions::InitConditions()
    : pimpl(new InitConditionsImpl)
{
}

void InitConditions::init(const OptionsPtr &)
{
    // TBD
}

boost::shared_ptr<vector<float> > InitConditions::field() const
{
    return pimpl->field;
}
