#!/usr/bin/env bash
# MERCIER Wilfried - LAM <wilfried.mercier@lam.fr>
# Automatically run tests and coverage report

function error_message () {
   echo -e "To generate a text-based report in the terminal, run\n\t\x1b[1m./test_and_coverage\x1b[0m\n"
   echo -e "Or, to generate a web-based report in the browser, run\n\t\x1b[1m./test_and_coverage html\x1b[0m"
}

# Handling potential errors in arguments
if [ "$#" -gt 1 ]
then
   echo -e "\x1b[31mWrong number of arguments provided (""$#"")\x1b[0m\n"
   error_message
   exit 1
elif [ "$#" -eq 1 ] && [ "$1" != "html" ]
then
   echo -e "\x1b[31mWrong argument provided (""$1"")\x1b[0m\n"
   error_message
   exit 2
fi

# Running coverage and tests
coverage run -m pytest -v -s

if [ "$#" -eq 0 ]
then
   coverage report -m
else
   coverage html
fi

exit 0
