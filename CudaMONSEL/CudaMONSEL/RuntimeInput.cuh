#ifndef _RUNTIME_INPUT_CUH_
#define _RUNTIME_INPUT_CUH_

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace RuntimeInput
{
   enum JsonType
   {
      JSON_NULL,
      JSON_BOOL,
      JSON_NUMBER,
      JSON_STRING,
      JSON_ARRAY,
      JSON_OBJECT
   };

   class JsonValue
   {
   public:
      JsonType type;
      bool boolValue;
      double numberValue;
      std::string stringValue;
      std::vector<JsonValue> arrayValue;
      std::map<std::string, JsonValue> objectValue;

      JsonValue() : type(JSON_NULL), boolValue(false), numberValue(0.0) {}

      static JsonValue makeBool(bool value)
      {
         JsonValue result;
         result.type = JSON_BOOL;
         result.boolValue = value;
         return result;
      }

      static JsonValue makeNumber(double value)
      {
         JsonValue result;
         result.type = JSON_NUMBER;
         result.numberValue = value;
         return result;
      }

      static JsonValue makeString(const std::string& value)
      {
         JsonValue result;
         result.type = JSON_STRING;
         result.stringValue = value;
         return result;
      }

      static JsonValue makeArray()
      {
         JsonValue result;
         result.type = JSON_ARRAY;
         return result;
      }

      static JsonValue makeObject()
      {
         JsonValue result;
         result.type = JSON_OBJECT;
         return result;
      }

      bool isObject() const { return type == JSON_OBJECT; }
      bool isArray() const { return type == JSON_ARRAY; }
      bool isString() const { return type == JSON_STRING; }
      bool isNumber() const { return type == JSON_NUMBER; }
      bool isBool() const { return type == JSON_BOOL; }

      const JsonValue* find(const std::string& key) const
      {
         if (!isObject()) return nullptr;
         std::map<std::string, JsonValue>::const_iterator it = objectValue.find(key);
         return it == objectValue.end() ? nullptr : &it->second;
      }

      const JsonValue& require(const std::string& key) const
      {
         const JsonValue* value = find(key);
         if (value == nullptr) throw std::runtime_error("Missing required JSON field: " + key);
         return *value;
      }

      std::string stringOr(const std::string& key, const std::string& defaultValue) const
      {
         const JsonValue* value = find(key);
         if (value == nullptr) return defaultValue;
         if (!value->isString()) throw std::runtime_error("JSON field must be a string: " + key);
         return value->stringValue;
      }

      double numberOr(const std::string& key, double defaultValue) const
      {
         const JsonValue* value = find(key);
         if (value == nullptr) return defaultValue;
         if (!value->isNumber()) throw std::runtime_error("JSON field must be a number: " + key);
         return value->numberValue;
      }

      int intOr(const std::string& key, int defaultValue) const
      {
         return (int)numberOr(key, (double)defaultValue);
      }

      bool boolOr(const std::string& key, bool defaultValue) const
      {
         const JsonValue* value = find(key);
         if (value == nullptr) return defaultValue;
         if (!value->isBool()) throw std::runtime_error("JSON field must be a boolean: " + key);
         return value->boolValue;
      }
   };

   class JsonParser
   {
   public:
      explicit JsonParser(const std::string& text) : mText(text), mPos(0) {}

      JsonValue parse()
      {
         JsonValue value = parseValue();
         skipWhitespace();
         if (mPos != mText.size()) error("Unexpected trailing content");
         return value;
      }

   private:
      const std::string& mText;
      size_t mPos;

      void skipWhitespace()
      {
         while (mPos < mText.size() && std::isspace((unsigned char)mText[mPos])) ++mPos;
      }

      bool consume(char ch)
      {
         skipWhitespace();
         if (mPos < mText.size() && mText[mPos] == ch) {
            ++mPos;
            return true;
         }
         return false;
      }

      void expect(char ch)
      {
         if (!consume(ch)) {
            std::string message("Expected '");
            message += ch;
            message += "'";
            error(message);
         }
      }

      void error(const std::string& message) const
      {
         std::ostringstream out;
         out << "JSON parse error at byte " << mPos << ": " << message;
         throw std::runtime_error(out.str());
      }

      JsonValue parseValue()
      {
         skipWhitespace();
         if (mPos >= mText.size()) error("Unexpected end of file");

         char ch = mText[mPos];
         if (ch == '{') return parseObject();
         if (ch == '[') return parseArray();
         if (ch == '"') return JsonValue::makeString(parseString());
         if (ch == '-' || (ch >= '0' && ch <= '9')) return parseNumber();
         if (matchLiteral("true")) return JsonValue::makeBool(true);
         if (matchLiteral("false")) return JsonValue::makeBool(false);
         if (matchLiteral("null")) return JsonValue();

         error("Unexpected token");
         return JsonValue();
      }

      bool matchLiteral(const char* literal)
      {
         size_t start = mPos;
         for (size_t i = 0; literal[i] != '\0'; ++i) {
            if (start + i >= mText.size() || mText[start + i] != literal[i]) return false;
         }
         mPos += std::strlen(literal);
         return true;
      }

      JsonValue parseObject()
      {
         JsonValue result = JsonValue::makeObject();
         expect('{');
         if (consume('}')) return result;

         while (true) {
            skipWhitespace();
            if (mPos >= mText.size() || mText[mPos] != '"') error("Expected object key");
            std::string key = parseString();
            expect(':');
            result.objectValue[key] = parseValue();
            if (consume('}')) break;
            expect(',');
         }
         return result;
      }

      JsonValue parseArray()
      {
         JsonValue result = JsonValue::makeArray();
         expect('[');
         if (consume(']')) return result;

         while (true) {
            result.arrayValue.push_back(parseValue());
            if (consume(']')) break;
            expect(',');
         }
         return result;
      }

      std::string parseString()
      {
         expect('"');
         std::string result;
         while (mPos < mText.size()) {
            char ch = mText[mPos++];
            if (ch == '"') return result;
            if (ch == '\\') {
               if (mPos >= mText.size()) error("Unterminated escape sequence");
               char escaped = mText[mPos++];
               switch (escaped) {
               case '"': result += '"'; break;
               case '\\': result += '\\'; break;
               case '/': result += '/'; break;
               case 'b': result += '\b'; break;
               case 'f': result += '\f'; break;
               case 'n': result += '\n'; break;
               case 'r': result += '\r'; break;
               case 't': result += '\t'; break;
               case 'u':
                  error("Unicode escapes are not supported in runtime JSON input");
                  break;
               default:
                  error("Invalid escape sequence");
               }
            }
            else {
               result += ch;
            }
         }
         error("Unterminated string");
         return result;
      }

      JsonValue parseNumber()
      {
         const char* start = mText.c_str() + mPos;
         char* end = nullptr;
         double value = std::strtod(start, &end);
         if (end == start) error("Invalid number");
         mPos += (size_t)(end - start);
         return JsonValue::makeNumber(value);
      }
   };

   inline JsonValue parseText(const std::string& text)
   {
      JsonParser parser(text);
      return parser.parse();
   }

   inline JsonValue parseFile(const std::string& path)
   {
      std::ifstream file(path.c_str(), std::ios::in | std::ios::binary);
      if (!file.good()) throw std::runtime_error("Unable to open JSON input file: " + path);
      std::ostringstream buffer;
      buffer << file.rdbuf();
      return parseText(buffer.str());
   }
}

#endif
