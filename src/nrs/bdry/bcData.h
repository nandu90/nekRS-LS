inline char *_strcpy(char *destination, const char *source)
{
  char *original_dest = destination;
  while (*source != '\0') {
    *destination = *source;
    destination++;
    source++;
  }

  *destination = '\0';
  return original_dest;
}

inline char _toLower(char c)
{
  if (c >= 'A' && c <= 'Z') {
    return c + ('a' - 'A');
  }
  return c;
}

inline int strCompare(const char *str1, const char *str2)
{
  while (*str1 != '\0' && *str2 != '\0' && (_toLower(*str1) == _toLower(*str2))) {
    str1++;
    str2++;
  }

  return _toLower(*str1) - _toLower(*str2);
}

struct bcData {
  char fieldName[32];

  int idM;

  int fieldOffset;
  int id;

  double time;
  dfloat x, y, z;
  dfloat nx, ny, nz;

  // tangential directions
  dfloat t1x, t1y, t1z;
  dfloat t2x, t2y, t2z;

  dfloat tr1, tr2;

  dfloat u, v, w;
  dfloat p;

  // interpolated velocity values
  dfloat uinterp, vinterp, winterp;

  int scalarId;
  dfloat s, flux;

  // interpolated scalar value
  dfloat sinterp;

  dfloat meshu, meshv, meshw;

  // properties
  dfloat trans, diff;

  @globalPtr const dfloat *usrwrk;
};
