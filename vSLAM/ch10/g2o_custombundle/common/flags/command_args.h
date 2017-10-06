
#ifndef G2O_COMMAND_ARGS_H
#define G2O_COMMAND_ARGS_H

#include <string>
#include <vector>
#include <iostream>
#include <sstream>


/**
 * \brief Command line parsing of argc and argv.
 *
 * Parse the command line to get the program options. Additionally,
 * we can store the config in a file and reload a parameter set from
 * this file.
 */
class CommandArgs
{
  public:
    struct CommandArgument
    {
      std::string name;
      std::string description;
      int type;
      void* data;
      bool parsed;
      bool optional;
      CommandArgument() : name(""), description(""), type(0), data(0), parsed(false), optional(false)
      {}
    };
  public:
    CommandArgs();
    virtual ~CommandArgs();

    /**
     * parse the command line for the requested parameters.
     * @param argc the number of params
     * @param argv the value array
     * @param exitOnError call exit() if the parsing fails
     * @return true, if parsing was correct
     */
    bool parseArgs(int argc, char** argv, bool exitOnError = true);

    /** add a bool parameter, if found on the command line, will toggle defValue */
    void param(const std::string& name, bool& p, bool defValue, const std::string& desc);
    /** add a int parameter */
    void param(const std::string& name, int& p, int defValue, const std::string& desc);
    /** add a float parameter */
    void param(const std::string& name, float& p, float defValue, const std::string& desc);
    /** add a float parameter */
    void param(const std::string& name, double& p, double defValue, const std::string& desc);
    /** add a string parameter */
    void param(const std::string& name, std::string& p, const std::string& defValue, const std::string& desc);
    /** add an int vector parameter */
    void param(const std::string& name, std::vector<int>& p, const std::vector<int>& defValue, const std::string& desc);
    /** add an vector of doubles as a parameter */
    void param(const std::string& name, std::vector<double>& p, const std::vector<double>& defValue, const std::string& desc);
    /** add a param wich is specified as a plain argument */
    void paramLeftOver(const std::string& name, std::string& p, const std::string& defValue, const std::string& desc, bool optional = false);

    /**
     * print the value of all params to an ostream
     */
    void printParams(std::ostream& os);

    //! return the banner string
    const std::string& getBanner() const { return _banner; }
    void setBanner(const std::string& banner);

    /**
     * print the help
     */
    void printHelp(std::ostream& os);

    /**
     * returns true, if the param was parsed via the command line
     */
    bool parsedParam(const std::string& paramFlag) const;

  protected:
    std::vector<CommandArgument> _args;
    std::vector<CommandArgument> _leftOvers;
    std::vector<CommandArgument> _leftOversOptional;
    std::string _banner;
    std::string _progName;

    const char* type2str(int t) const;
    void str2arg(const std::string& input, CommandArgument& ca) const;
    std::string arg2str(const CommandArgument& ca) const;

    std::string trim(const std::string& s) const;
};


#endif
