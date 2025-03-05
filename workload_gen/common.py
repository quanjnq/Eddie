import json
from sqlalchemy.orm import sessionmaker, aliased
from sqlalchemy.dialects import postgresql, mysql, sqlite
from sqlalchemy import create_engine, Index, func, cast, desc, asc, CheckConstraint, and_, or_, any_, all_
from sqlalchemy import text as txt
from sqlalchemy import update, delete, select, subquery, exists, tuple_
from sqlalchemy import Table, Column, Index, Integer, BigInteger, Boolean, MetaData, ForeignKey, Float, REAL, Text, CHAR, Unicode, NUMERIC
from sqlalchemy.schema import CreateTable, CreateIndex, DropTable
from sqlalchemy.engine.default import DefaultDialect
from sqlalchemy.sql.sqltypes import NullType, String, DateTime, Date
from sqlalchemy import distinct

import pprint
pp = pprint.PrettyPrinter(indent=4)

# python2/3 compatible.
PY3 = str is not bytes
text = str
int_type = int
str_type = str

TMP_QUERY = "/tmp/query"
CONJ = ["and", "or", "and", "and", "and", "and", "and", "and"] # Increase the AND probability
OP = ['+', '-', '*', '/']
COMP = ['<', '<=', '<>', '>', '>=', '==', '=']
FUNC_LIST = [func.max, func.min, func.count]
FUNC_LIST_NUMBER = [func.sum, func.max, func.min,
                    func.avg, func.count, func.variance]

FUNC_INDEX = [func.ltrim, func.rtrim, func.length]
FUNC_DATETIME_LIST = [func.quote]
# now we only perform cast between number types
CAST_TO_TYPE = [Integer, Float, Boolean]


class StringLiteral(String):
    """Teach SA how to literalize various things."""

    def literal_processor(self, dialect):
        super_processor = super(StringLiteral, self).literal_processor(dialect)

        def process(value):
            if isinstance(value, int_type):
                return text(value)
            if not isinstance(value, str_type):
                value = text(value)
            result = super_processor(value)
            if isinstance(result, bytes):
                result = result.decode(dialect.encoding)
            return result
        return process


class LiteralDialect(DefaultDialect):
    colspecs = {
        # prevent various encoding explosions
        String: StringLiteral,
        # teach SA about how to literalize a datetime
        DateTime: StringLiteral,
        Date: StringLiteral,

        # don't format py2 long integers to NULL
        NullType: StringLiteral,
    }


def ret_typename_from_class(typeclass):
    
    if "VARCHAR" in str(typeclass):
        typename = "String"
    # add support for types read from TPCH:
    elif "INTEGER" in str(typeclass) or "OID" in str(typeclass):
        typename = "Integer"
    elif "CHAR" in str(typeclass):
        typename = "String"
    elif "NUMERIC" in str(typeclass) or "FLOAT" in str(typeclass):
        typename = "Float"
    elif "DATE" in str(typeclass) or "TIME" in str(typeclass):
        typename = "DateTime"
    elif "BOOLEAN" in str(typeclass):
        typename = "Boolean"
    elif "TIMESTAMP" in str(typeclass):
        typename = "DateTime"
    elif "TEXT" in str(typeclass):
        typename = "Text"
    elif "SMALLINT" in str(typeclass):
        typename = "SmallInt"
    elif "ARRAY" in str(typeclass):
        typename = "Array"
    elif "TSVECTOR" in str(typeclass):
        typename = "TSVector"
    elif "BYTEA" in str(typeclass):
        typename = "ByteA"
    elif "DOUBLE_PRECISION" in str(typeclass):
        typename = "DOUBLE_PRECISION"
    elif "BIGINT" in str(typeclass):
        typename = "BIGINT"
    else:
        typename = typeclass.__name__
    return typename


def literalquery(statement, db="postgres"):
    import sqlalchemy.orm
    if isinstance(statement, sqlalchemy.orm.Query):
        statement = statement.statement

    if db == "myqsl":
        target_dialect = mysql.dialect()
    elif db == "postgres":
        target_dialect = LiteralDialect()
    elif db == "sqlite":
        target_dialect = sqlite.dialect()
    else:
        target_dialect = LiteralDialect()

    return statement.compile(dialect=target_dialect,
                             compile_kwargs={'literal_binds': True},).string


def get_compatible_function(column):
    if (ret_typename_from_class(column.type) in ["Integer", "Float"]):
        return FUNC_LIST_NUMBER
    return FUNC_LIST


def get_selectable_column(alc_selectable, col_backlist=None):
    # return a list of columns in selectable: subquery or simple_table
    column_list = []
    try:
        for item in alc_selectable.columns:
            if item.name not in col_backlist:
                column_list.append(item)
        return column_list
    except:
        # has bug in sqlalchemy
        return column_list


def get_compatible_column(selectable_columns, target_type):
    number_type = ["FLOAT", "INT"]
    string_type = "CHAR"
    selectable_type = []
    for c in selectable_columns:
        if str(c.type) == target_type:
            return c
        elif str(c.type) in number_type and target_type in number_type:
            return c
        elif "NUMERIC" in target_type and str(c.type) in number_type:
            return c
        elif string_type in str(c.type) and string_type in target_type:
            return c
        selectable_type.append(str(c.type))

def load_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
        return data
