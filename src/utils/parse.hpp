// example.hpp
#include <boost/fusion/adapted/std_tuple.hpp>
#include <boost/spirit/home/x3.hpp>
#include <iostream>

namespace x3 = boost::spirit::x3;

auto parse = [](auto &b, auto e, auto const &p, auto &...binds)
{
  auto attr = std::tie(binds...);
  return x3::phrase_parse(b, e, p, x3::space, attr);
};

inline size_t fast_parse(auto f, auto l, float *mz, float *intensity)
{
  return parse(f, l, x3::float_ >> x3::float_, *mz, *intensity);
}

inline int fast_str_compare(const char *ptr0, const char *ptr1, int len)
{
  int fast = len / sizeof(size_t) + 1;
  int offset = (fast - 1) * sizeof(size_t);
  int current_block = 0;

  if (len <= sizeof(size_t))
  {
    fast = 0;
  }

  size_t *lptr0 = (size_t *)ptr0;
  size_t *lptr1 = (size_t *)ptr1;

  while (current_block < fast)
  {
    if ((lptr0[current_block] ^ lptr1[current_block]))
    {
      int pos;
      for (pos = current_block * sizeof(size_t); pos < len; ++pos)
      {
        if ((ptr0[pos] ^ ptr1[pos]) || (ptr0[pos] == 0) || (ptr1[pos] == 0))
        {
          return (int)((unsigned char)ptr0[pos] - (unsigned char)ptr1[pos]);
        }
      }
    }

    ++current_block;
  }

  while (len > offset)
  {
    if ((ptr0[offset] ^ ptr1[offset]))
    {
      return (int)((unsigned char)ptr0[offset] - (unsigned char)ptr1[offset]);
    }
    ++offset;
  }

  return 0;
}
