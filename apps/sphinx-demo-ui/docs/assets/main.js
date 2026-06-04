/* v8 ignore next */ /* v8 ignore next */ 'use strict'; /* v8 ignore next */ /* v8 ignore next */
'use strict'; /* v8 ignore next */ /* v8 ignore next */
(() => { /* v8 ignore next */ /* v8 ignore next */
  var Ce = Object.create; /* v8 ignore next */ /* v8 ignore next */
  var ne = Object.defineProperty; /* v8 ignore next */ /* v8 ignore next */
  var Pe = Object.getOwnPropertyDescriptor; /* v8 ignore next */ /* v8 ignore next */
  var Oe = Object.getOwnPropertyNames; /* v8 ignore next */ /* v8 ignore next */
  var _e = Object.getPrototypeOf, /* v8 ignore next */ /* v8 ignore next */
    Re = Object.prototype.hasOwnProperty; /* v8 ignore next */ /* v8 ignore next */
  var Me = (t, e) => () => (e || t((e = { exports: {} }).exports, e), e.exports); /* v8 ignore next */ /* v8 ignore next */
  var Fe = (t, e, n, r) => { /* v8 ignore next */ /* v8 ignore next */
    if ((e && typeof e == 'object') || typeof e == 'function') /* v8 ignore next */ /* v8 ignore next */
      for (let i of Oe(e)) /* v8 ignore next */ /* v8 ignore next */
        !Re.call(t, i) && /* v8 ignore next */ /* v8 ignore next */
          i !== n && /* v8 ignore next */ /* v8 ignore next */
          ne(t, i, { get: () => e[i], enumerable: !(r = Pe(e, i)) || r.enumerable }); /* v8 ignore next */ /* v8 ignore next */
    return t; /* v8 ignore next */ /* v8 ignore next */
  }; /* v8 ignore next */ /* v8 ignore next */
  var De = (t, e, n) => ( /* v8 ignore next */ /* v8 ignore next */
    (n = t != null ? Ce(_e(t)) : {}), /* v8 ignore next */ /* v8 ignore next */
    Fe(e || !t || !t.__esModule ? ne(n, 'default', { value: t, enumerable: !0 }) : n, t) /* v8 ignore next */ /* v8 ignore next */
  ); /* v8 ignore next */ /* v8 ignore next */
  var ae = Me((se, oe) => { /* v8 ignore next */ /* v8 ignore next */
    (function () { /* v8 ignore next */ /* v8 ignore next */
      var t = function (e) { /* v8 ignore next */ /* v8 ignore next */
        var n = new t.Builder(); /* v8 ignore next */ /* v8 ignore next */
        return ( /* v8 ignore next */ /* v8 ignore next */
          n.pipeline.add(t.trimmer, t.stopWordFilter, t.stemmer), /* v8 ignore next */ /* v8 ignore next */
          n.searchPipeline.add(t.stemmer), /* v8 ignore next */ /* v8 ignore next */
          e.call(n, n), /* v8 ignore next */ /* v8 ignore next */
          n.build() /* v8 ignore next */ /* v8 ignore next */
        ); /* v8 ignore next */ /* v8 ignore next */
      }; /* v8 ignore next */ /* v8 ignore next */
      t.version = '2.3.9'; /* v8 ignore next */ /* v8 ignore next */
      ((t.utils = {}), /* v8 ignore next */ /* v8 ignore next */
        (t.utils.warn = (function (e) { /* v8 ignore next */ /* v8 ignore next */
          return function (n) { /* v8 ignore next */ /* v8 ignore next */
            e.console && console.warn && console.warn(n); /* v8 ignore next */ /* v8 ignore next */
          }; /* v8 ignore next */ /* v8 ignore next */
        })(this)), /* v8 ignore next */ /* v8 ignore next */
        (t.utils.asString = function (e) { /* v8 ignore next */ /* v8 ignore next */
          return e == null ? '' : e.toString(); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.utils.clone = function (e) { /* v8 ignore next */ /* v8 ignore next */
          if (e == null) return e; /* v8 ignore next */ /* v8 ignore next */
          for (var n = Object.create(null), r = Object.keys(e), i = 0; i < r.length; i++) { /* v8 ignore next */ /* v8 ignore next */
            var s = r[i], /* v8 ignore next */ /* v8 ignore next */
              o = e[s]; /* v8 ignore next */ /* v8 ignore next */
            if (Array.isArray(o)) { /* v8 ignore next */ /* v8 ignore next */
              n[s] = o.slice(); /* v8 ignore next */ /* v8 ignore next */
              continue; /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
            if (typeof o == 'string' || typeof o == 'number' || typeof o == 'boolean') { /* v8 ignore next */ /* v8 ignore next */
              n[s] = o; /* v8 ignore next */ /* v8 ignore next */
              continue; /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
            throw new TypeError('clone is not deep and does not support nested objects'); /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
          return n; /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.FieldRef = function (e, n, r) { /* v8 ignore next */ /* v8 ignore next */
          ((this.docRef = e), (this.fieldName = n), (this._stringValue = r)); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.FieldRef.joiner = '/'), /* v8 ignore next */ /* v8 ignore next */
        (t.FieldRef.fromString = function (e) { /* v8 ignore next */ /* v8 ignore next */
          var n = e.indexOf(t.FieldRef.joiner); /* v8 ignore next */ /* v8 ignore next */
          if (n === -1) throw 'malformed field ref string'; /* v8 ignore next */ /* v8 ignore next */
          var r = e.slice(0, n), /* v8 ignore next */ /* v8 ignore next */
            i = e.slice(n + 1); /* v8 ignore next */ /* v8 ignore next */
          return new t.FieldRef(i, r, e); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.FieldRef.prototype.toString = function () { /* v8 ignore next */ /* v8 ignore next */
          return ( /* v8 ignore next */ /* v8 ignore next */
            this._stringValue == null && /* v8 ignore next */ /* v8 ignore next */
              (this._stringValue = this.fieldName + t.FieldRef.joiner + this.docRef), /* v8 ignore next */ /* v8 ignore next */
            this._stringValue /* v8 ignore next */ /* v8 ignore next */
          ); /* v8 ignore next */ /* v8 ignore next */
        })); /* v8 ignore next */ /* v8 ignore next */
      ((t.Set = function (e) { /* v8 ignore next */ /* v8 ignore next */
        if (((this.elements = Object.create(null)), e)) { /* v8 ignore next */ /* v8 ignore next */
          this.length = e.length; /* v8 ignore next */ /* v8 ignore next */
          for (var n = 0; n < this.length; n++) this.elements[e[n]] = !0; /* v8 ignore next */ /* v8 ignore next */
        } else this.length = 0; /* v8 ignore next */ /* v8 ignore next */
      }), /* v8 ignore next */ /* v8 ignore next */
        (t.Set.complete = { /* v8 ignore next */ /* v8 ignore next */
          intersect: function (e) { /* v8 ignore next */ /* v8 ignore next */
            return e; /* v8 ignore next */ /* v8 ignore next */
          }, /* v8 ignore next */ /* v8 ignore next */
          union: function () { /* v8 ignore next */ /* v8 ignore next */
            return this; /* v8 ignore next */ /* v8 ignore next */
          }, /* v8 ignore next */ /* v8 ignore next */
          contains: function () { /* v8 ignore next */ /* v8 ignore next */
            return !0; /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.Set.empty = { /* v8 ignore next */ /* v8 ignore next */
          intersect: function () { /* v8 ignore next */ /* v8 ignore next */
            return this; /* v8 ignore next */ /* v8 ignore next */
          }, /* v8 ignore next */ /* v8 ignore next */
          union: function (e) { /* v8 ignore next */ /* v8 ignore next */
            return e; /* v8 ignore next */ /* v8 ignore next */
          }, /* v8 ignore next */ /* v8 ignore next */
          contains: function () { /* v8 ignore next */ /* v8 ignore next */
            return !1; /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.Set.prototype.contains = function (e) { /* v8 ignore next */ /* v8 ignore next */
          return !!this.elements[e]; /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.Set.prototype.intersect = function (e) { /* v8 ignore next */ /* v8 ignore next */
          var n, /* v8 ignore next */ /* v8 ignore next */
            r, /* v8 ignore next */ /* v8 ignore next */
            i, /* v8 ignore next */ /* v8 ignore next */
            s = []; /* v8 ignore next */ /* v8 ignore next */
          if (e === t.Set.complete) return this; /* v8 ignore next */ /* v8 ignore next */
          if (e === t.Set.empty) return e; /* v8 ignore next */ /* v8 ignore next */
          (this.length < e.length ? ((n = this), (r = e)) : ((n = e), (r = this)), /* v8 ignore next */ /* v8 ignore next */
            (i = Object.keys(n.elements))); /* v8 ignore next */ /* v8 ignore next */
          for (var o = 0; o < i.length; o++) { /* v8 ignore next */ /* v8 ignore next */
            var a = i[o]; /* v8 ignore next */ /* v8 ignore next */
            a in r.elements && s.push(a); /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
          return new t.Set(s); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.Set.prototype.union = function (e) { /* v8 ignore next */ /* v8 ignore next */
          return e === t.Set.complete /* v8 ignore next */ /* v8 ignore next */
            ? t.Set.complete /* v8 ignore next */ /* v8 ignore next */
            : e === t.Set.empty /* v8 ignore next */ /* v8 ignore next */
              ? this /* v8 ignore next */ /* v8 ignore next */
              : new t.Set(Object.keys(this.elements).concat(Object.keys(e.elements))); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.idf = function (e, n) { /* v8 ignore next */ /* v8 ignore next */
          var r = 0; /* v8 ignore next */ /* v8 ignore next */
          for (var i in e) i != '_index' && (r += Object.keys(e[i]).length); /* v8 ignore next */ /* v8 ignore next */
          var s = (n - r + 0.5) / (r + 0.5); /* v8 ignore next */ /* v8 ignore next */
          return Math.log(1 + Math.abs(s)); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.Token = function (e, n) { /* v8 ignore next */ /* v8 ignore next */
          ((this.str = e || ''), (this.metadata = n || {})); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.Token.prototype.toString = function () { /* v8 ignore next */ /* v8 ignore next */
          return this.str; /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.Token.prototype.update = function (e) { /* v8 ignore next */ /* v8 ignore next */
          return ((this.str = e(this.str, this.metadata)), this); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.Token.prototype.clone = function (e) { /* v8 ignore next */ /* v8 ignore next */
          return ( /* v8 ignore next */ /* v8 ignore next */
            (e = /* v8 ignore next */ /* v8 ignore next */
              e || /* v8 ignore next */ /* v8 ignore next */
              function (n) { /* v8 ignore next */ /* v8 ignore next */
                return n; /* v8 ignore next */ /* v8 ignore next */
              }), /* v8 ignore next */ /* v8 ignore next */
            new t.Token(e(this.str, this.metadata), this.metadata) /* v8 ignore next */ /* v8 ignore next */
          ); /* v8 ignore next */ /* v8 ignore next */
        })); /* v8 ignore next */ /* v8 ignore next */
      ((t.tokenizer = function (e, n) { /* v8 ignore next */ /* v8 ignore next */
        if (e == null || e == null) return []; /* v8 ignore next */ /* v8 ignore next */
        if (Array.isArray(e)) /* v8 ignore next */ /* v8 ignore next */
          return e.map(function (y) { /* v8 ignore next */ /* v8 ignore next */
            return new t.Token(t.utils.asString(y).toLowerCase(), t.utils.clone(n)); /* v8 ignore next */ /* v8 ignore next */
          }); /* v8 ignore next */ /* v8 ignore next */
        for (var r = e.toString().toLowerCase(), i = r.length, s = [], o = 0, a = 0; o <= i; o++) { /* v8 ignore next */ /* v8 ignore next */
          var l = r.charAt(o), /* v8 ignore next */ /* v8 ignore next */
            u = o - a; /* v8 ignore next */ /* v8 ignore next */
          if (l.match(t.tokenizer.separator) || o == i) { /* v8 ignore next */ /* v8 ignore next */
            if (u > 0) { /* v8 ignore next */ /* v8 ignore next */
              var d = t.utils.clone(n) || {}; /* v8 ignore next */ /* v8 ignore next */
              ((d.position = [a, u]), (d.index = s.length), s.push(new t.Token(r.slice(a, o), d))); /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
            a = o + 1; /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
        return s; /* v8 ignore next */ /* v8 ignore next */
      }), /* v8 ignore next */ /* v8 ignore next */
        (t.tokenizer.separator = /[\s\-]+/)); /* v8 ignore next */ /* v8 ignore next */
      ((t.Pipeline = function () { /* v8 ignore next */ /* v8 ignore next */
        this._stack = []; /* v8 ignore next */ /* v8 ignore next */
      }), /* v8 ignore next */ /* v8 ignore next */
        (t.Pipeline.registeredFunctions = Object.create(null)), /* v8 ignore next */ /* v8 ignore next */
        (t.Pipeline.registerFunction = function (e, n) { /* v8 ignore next */ /* v8 ignore next */
          (n in this.registeredFunctions && /* v8 ignore next */ /* v8 ignore next */
            t.utils.warn('Overwriting existing registered function: ' + n), /* v8 ignore next */ /* v8 ignore next */
            (e.label = n), /* v8 ignore next */ /* v8 ignore next */
            (t.Pipeline.registeredFunctions[e.label] = e)); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.Pipeline.warnIfFunctionNotRegistered = function (e) { /* v8 ignore next */ /* v8 ignore next */
          var n = e.label && e.label in this.registeredFunctions; /* v8 ignore next */ /* v8 ignore next */
          n || /* v8 ignore next */ /* v8 ignore next */
            t.utils.warn( /* v8 ignore next */ /* v8 ignore next */
              `Function is not registered with pipeline. This may cause problems when serialising the index. /* v8 ignore next */ /* v8 ignore next */
`, /* v8 ignore next */ /* v8 ignore next */
              e /* v8 ignore next */ /* v8 ignore next */
            ); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.Pipeline.load = function (e) { /* v8 ignore next */ /* v8 ignore next */
          var n = new t.Pipeline(); /* v8 ignore next */ /* v8 ignore next */
          return ( /* v8 ignore next */ /* v8 ignore next */
            e.forEach(function (r) { /* v8 ignore next */ /* v8 ignore next */
              var i = t.Pipeline.registeredFunctions[r]; /* v8 ignore next */ /* v8 ignore next */
              if (i) n.add(i); /* v8 ignore next */ /* v8 ignore next */
              else throw new Error('Cannot load unregistered function: ' + r); /* v8 ignore next */ /* v8 ignore next */
            }), /* v8 ignore next */ /* v8 ignore next */
            n /* v8 ignore next */ /* v8 ignore next */
          ); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.Pipeline.prototype.add = function () { /* v8 ignore next */ /* v8 ignore next */
          var e = Array.prototype.slice.call(arguments); /* v8 ignore next */ /* v8 ignore next */
          e.forEach(function (n) { /* v8 ignore next */ /* v8 ignore next */
            (t.Pipeline.warnIfFunctionNotRegistered(n), this._stack.push(n)); /* v8 ignore next */ /* v8 ignore next */
          }, this); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.Pipeline.prototype.after = function (e, n) { /* v8 ignore next */ /* v8 ignore next */
          t.Pipeline.warnIfFunctionNotRegistered(n); /* v8 ignore next */ /* v8 ignore next */
          var r = this._stack.indexOf(e); /* v8 ignore next */ /* v8 ignore next */
          if (r == -1) throw new Error('Cannot find existingFn'); /* v8 ignore next */ /* v8 ignore next */
          ((r = r + 1), this._stack.splice(r, 0, n)); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.Pipeline.prototype.before = function (e, n) { /* v8 ignore next */ /* v8 ignore next */
          t.Pipeline.warnIfFunctionNotRegistered(n); /* v8 ignore next */ /* v8 ignore next */
          var r = this._stack.indexOf(e); /* v8 ignore next */ /* v8 ignore next */
          if (r == -1) throw new Error('Cannot find existingFn'); /* v8 ignore next */ /* v8 ignore next */
          this._stack.splice(r, 0, n); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.Pipeline.prototype.remove = function (e) { /* v8 ignore next */ /* v8 ignore next */
          var n = this._stack.indexOf(e); /* v8 ignore next */ /* v8 ignore next */
          n != -1 && this._stack.splice(n, 1); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.Pipeline.prototype.run = function (e) { /* v8 ignore next */ /* v8 ignore next */
          for (var n = this._stack.length, r = 0; r < n; r++) { /* v8 ignore next */ /* v8 ignore next */
            for (var i = this._stack[r], s = [], o = 0; o < e.length; o++) { /* v8 ignore next */ /* v8 ignore next */
              var a = i(e[o], o, e); /* v8 ignore next */ /* v8 ignore next */
              if (!(a == null || a === '')) /* v8 ignore next */ /* v8 ignore next */
                if (Array.isArray(a)) for (var l = 0; l < a.length; l++) s.push(a[l]); /* v8 ignore next */ /* v8 ignore next */
                else s.push(a); /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
            e = s; /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
          return e; /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.Pipeline.prototype.runString = function (e, n) { /* v8 ignore next */ /* v8 ignore next */
          var r = new t.Token(e, n); /* v8 ignore next */ /* v8 ignore next */
          return this.run([r]).map(function (i) { /* v8 ignore next */ /* v8 ignore next */
            return i.toString(); /* v8 ignore next */ /* v8 ignore next */
          }); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.Pipeline.prototype.reset = function () { /* v8 ignore next */ /* v8 ignore next */
          this._stack = []; /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.Pipeline.prototype.toJSON = function () { /* v8 ignore next */ /* v8 ignore next */
          return this._stack.map(function (e) { /* v8 ignore next */ /* v8 ignore next */
            return (t.Pipeline.warnIfFunctionNotRegistered(e), e.label); /* v8 ignore next */ /* v8 ignore next */
          }); /* v8 ignore next */ /* v8 ignore next */
        })); /* v8 ignore next */ /* v8 ignore next */
      ((t.Vector = function (e) { /* v8 ignore next */ /* v8 ignore next */
        ((this._magnitude = 0), (this.elements = e || [])); /* v8 ignore next */ /* v8 ignore next */
      }), /* v8 ignore next */ /* v8 ignore next */
        (t.Vector.prototype.positionForIndex = function (e) { /* v8 ignore next */ /* v8 ignore next */
          if (this.elements.length == 0) return 0; /* v8 ignore next */ /* v8 ignore next */
          for ( /* v8 ignore next */ /* v8 ignore next */
            var n = 0, /* v8 ignore next */ /* v8 ignore next */
              r = this.elements.length / 2, /* v8 ignore next */ /* v8 ignore next */
              i = r - n, /* v8 ignore next */ /* v8 ignore next */
              s = Math.floor(i / 2), /* v8 ignore next */ /* v8 ignore next */
              o = this.elements[s * 2]; /* v8 ignore next */ /* v8 ignore next */
            i > 1 && (o < e && (n = s), o > e && (r = s), o != e); /* v8 ignore next */ /* v8 ignore next */
          ) /* v8 ignore next */ /* v8 ignore next */
            ((i = r - n), (s = n + Math.floor(i / 2)), (o = this.elements[s * 2])); /* v8 ignore next */ /* v8 ignore next */
          if (o == e || o > e) return s * 2; /* v8 ignore next */ /* v8 ignore next */
          if (o < e) return (s + 1) * 2; /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.Vector.prototype.insert = function (e, n) { /* v8 ignore next */ /* v8 ignore next */
          this.upsert(e, n, function () { /* v8 ignore next */ /* v8 ignore next */
            throw 'duplicate index'; /* v8 ignore next */ /* v8 ignore next */
          }); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.Vector.prototype.upsert = function (e, n, r) { /* v8 ignore next */ /* v8 ignore next */
          this._magnitude = 0; /* v8 ignore next */ /* v8 ignore next */
          var i = this.positionForIndex(e); /* v8 ignore next */ /* v8 ignore next */
          this.elements[i] == e /* v8 ignore next */ /* v8 ignore next */
            ? (this.elements[i + 1] = r(this.elements[i + 1], n)) /* v8 ignore next */ /* v8 ignore next */
            : this.elements.splice(i, 0, e, n); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.Vector.prototype.magnitude = function () { /* v8 ignore next */ /* v8 ignore next */
          if (this._magnitude) return this._magnitude; /* v8 ignore next */ /* v8 ignore next */
          for (var e = 0, n = this.elements.length, r = 1; r < n; r += 2) { /* v8 ignore next */ /* v8 ignore next */
            var i = this.elements[r]; /* v8 ignore next */ /* v8 ignore next */
            e += i * i; /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
          return (this._magnitude = Math.sqrt(e)); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.Vector.prototype.dot = function (e) { /* v8 ignore next */ /* v8 ignore next */
          for ( /* v8 ignore next */ /* v8 ignore next */
            var n = 0, /* v8 ignore next */ /* v8 ignore next */
              r = this.elements, /* v8 ignore next */ /* v8 ignore next */
              i = e.elements, /* v8 ignore next */ /* v8 ignore next */
              s = r.length, /* v8 ignore next */ /* v8 ignore next */
              o = i.length, /* v8 ignore next */ /* v8 ignore next */
              a = 0, /* v8 ignore next */ /* v8 ignore next */
              l = 0, /* v8 ignore next */ /* v8 ignore next */
              u = 0, /* v8 ignore next */ /* v8 ignore next */
              d = 0; /* v8 ignore next */ /* v8 ignore next */
            u < s && d < o; /* v8 ignore next */ /* v8 ignore next */
          ) /* v8 ignore next */ /* v8 ignore next */
            ((a = r[u]), /* v8 ignore next */ /* v8 ignore next */
              (l = i[d]), /* v8 ignore next */ /* v8 ignore next */
              a < l /* v8 ignore next */ /* v8 ignore next */
                ? (u += 2) /* v8 ignore next */ /* v8 ignore next */
                : a > l /* v8 ignore next */ /* v8 ignore next */
                  ? (d += 2) /* v8 ignore next */ /* v8 ignore next */
                  : a == l && ((n += r[u + 1] * i[d + 1]), (u += 2), (d += 2))); /* v8 ignore next */ /* v8 ignore next */
          return n; /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.Vector.prototype.similarity = function (e) { /* v8 ignore next */ /* v8 ignore next */
          return this.dot(e) / this.magnitude() || 0; /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.Vector.prototype.toArray = function () { /* v8 ignore next */ /* v8 ignore next */
          for ( /* v8 ignore next */ /* v8 ignore next */
            var e = new Array(this.elements.length / 2), n = 1, r = 0; /* v8 ignore next */ /* v8 ignore next */
            n < this.elements.length; /* v8 ignore next */ /* v8 ignore next */
            n += 2, r++ /* v8 ignore next */ /* v8 ignore next */
          ) /* v8 ignore next */ /* v8 ignore next */
            e[r] = this.elements[n]; /* v8 ignore next */ /* v8 ignore next */
          return e; /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.Vector.prototype.toJSON = function () { /* v8 ignore next */ /* v8 ignore next */
          return this.elements; /* v8 ignore next */ /* v8 ignore next */
        })); /* v8 ignore next */ /* v8 ignore next */
      ((t.stemmer = (function () { /* v8 ignore next */ /* v8 ignore next */
        var e = { /* v8 ignore next */ /* v8 ignore next */
            ational: 'ate', /* v8 ignore next */ /* v8 ignore next */
            tional: 'tion', /* v8 ignore next */ /* v8 ignore next */
            enci: 'ence', /* v8 ignore next */ /* v8 ignore next */
            anci: 'ance', /* v8 ignore next */ /* v8 ignore next */
            izer: 'ize', /* v8 ignore next */ /* v8 ignore next */
            bli: 'ble', /* v8 ignore next */ /* v8 ignore next */
            alli: 'al', /* v8 ignore next */ /* v8 ignore next */
            entli: 'ent', /* v8 ignore next */ /* v8 ignore next */
            eli: 'e', /* v8 ignore next */ /* v8 ignore next */
            ousli: 'ous', /* v8 ignore next */ /* v8 ignore next */
            ization: 'ize', /* v8 ignore next */ /* v8 ignore next */
            ation: 'ate', /* v8 ignore next */ /* v8 ignore next */
            ator: 'ate', /* v8 ignore next */ /* v8 ignore next */
            alism: 'al', /* v8 ignore next */ /* v8 ignore next */
            iveness: 'ive', /* v8 ignore next */ /* v8 ignore next */
            fulness: 'ful', /* v8 ignore next */ /* v8 ignore next */
            ousness: 'ous', /* v8 ignore next */ /* v8 ignore next */
            aliti: 'al', /* v8 ignore next */ /* v8 ignore next */
            iviti: 'ive', /* v8 ignore next */ /* v8 ignore next */
            biliti: 'ble', /* v8 ignore next */ /* v8 ignore next */
            logi: 'log' /* v8 ignore next */ /* v8 ignore next */
          }, /* v8 ignore next */ /* v8 ignore next */
          n = { icate: 'ic', ative: '', alize: 'al', iciti: 'ic', ical: 'ic', ful: '', ness: '' }, /* v8 ignore next */ /* v8 ignore next */
          r = '[^aeiou]', /* v8 ignore next */ /* v8 ignore next */
          i = '[aeiouy]', /* v8 ignore next */ /* v8 ignore next */
          s = r + '[^aeiouy]*', /* v8 ignore next */ /* v8 ignore next */
          o = i + '[aeiou]*', /* v8 ignore next */ /* v8 ignore next */
          a = '^(' + s + ')?' + o + s, /* v8 ignore next */ /* v8 ignore next */
          l = '^(' + s + ')?' + o + s + '(' + o + ')?$', /* v8 ignore next */ /* v8 ignore next */
          u = '^(' + s + ')?' + o + s + o + s, /* v8 ignore next */ /* v8 ignore next */
          d = '^(' + s + ')?' + i, /* v8 ignore next */ /* v8 ignore next */
          y = new RegExp(a), /* v8 ignore next */ /* v8 ignore next */
          p = new RegExp(u), /* v8 ignore next */ /* v8 ignore next */
          b = new RegExp(l), /* v8 ignore next */ /* v8 ignore next */
          g = new RegExp(d), /* v8 ignore next */ /* v8 ignore next */
          L = /^(.+?)(ss|i)es$/, /* v8 ignore next */ /* v8 ignore next */
          f = /^(.+?)([^s])s$/, /* v8 ignore next */ /* v8 ignore next */
          m = /^(.+?)eed$/, /* v8 ignore next */ /* v8 ignore next */
          S = /^(.+?)(ed|ing)$/, /* v8 ignore next */ /* v8 ignore next */
          w = /.$/, /* v8 ignore next */ /* v8 ignore next */
          k = /(at|bl|iz)$/, /* v8 ignore next */ /* v8 ignore next */
          _ = new RegExp('([^aeiouylsz])\\1$'), /* v8 ignore next */ /* v8 ignore next */
          B = new RegExp('^' + s + i + '[^aeiouwxy]$'), /* v8 ignore next */ /* v8 ignore next */
          A = /^(.+?[^aeiou])y$/, /* v8 ignore next */ /* v8 ignore next */
          j = /* v8 ignore next */ /* v8 ignore next */
            /^(.+?)(ational|tional|enci|anci|izer|bli|alli|entli|eli|ousli|ization|ation|ator|alism|iveness|fulness|ousness|aliti|iviti|biliti|logi)$/, /* v8 ignore next */ /* v8 ignore next */
          $ = /^(.+?)(icate|ative|alize|iciti|ical|ful|ness)$/, /* v8 ignore next */ /* v8 ignore next */
          V = /^(.+?)(al|ance|ence|er|ic|able|ible|ant|ement|ment|ent|ou|ism|ate|iti|ous|ive|ize)$/, /* v8 ignore next */ /* v8 ignore next */
          q = /^(.+?)(s|t)(ion)$/, /* v8 ignore next */ /* v8 ignore next */
          C = /^(.+?)e$/, /* v8 ignore next */ /* v8 ignore next */
          z = /ll$/, /* v8 ignore next */ /* v8 ignore next */
          W = new RegExp('^' + s + i + '[^aeiouwxy]$'), /* v8 ignore next */ /* v8 ignore next */
          N = function (c) { /* v8 ignore next */ /* v8 ignore next */
            var v, P, T, h, x, O, M; /* v8 ignore next */ /* v8 ignore next */
            if (c.length < 3) return c; /* v8 ignore next */ /* v8 ignore next */
            if ( /* v8 ignore next */ /* v8 ignore next */
              ((T = c.substr(0, 1)), /* v8 ignore next */ /* v8 ignore next */
              T == 'y' && (c = T.toUpperCase() + c.substr(1)), /* v8 ignore next */ /* v8 ignore next */
              (h = L), /* v8 ignore next */ /* v8 ignore next */
              (x = f), /* v8 ignore next */ /* v8 ignore next */
              h.test(c) ? (c = c.replace(h, '$1$2')) : x.test(c) && (c = c.replace(x, '$1$2')), /* v8 ignore next */ /* v8 ignore next */
              (h = m), /* v8 ignore next */ /* v8 ignore next */
              (x = S), /* v8 ignore next */ /* v8 ignore next */
              h.test(c)) /* v8 ignore next */ /* v8 ignore next */
            ) { /* v8 ignore next */ /* v8 ignore next */
              var E = h.exec(c); /* v8 ignore next */ /* v8 ignore next */
              ((h = y), h.test(E[1]) && ((h = w), (c = c.replace(h, '')))); /* v8 ignore next */ /* v8 ignore next */
            } else if (x.test(c)) { /* v8 ignore next */ /* v8 ignore next */
              var E = x.exec(c); /* v8 ignore next */ /* v8 ignore next */
              ((v = E[1]), /* v8 ignore next */ /* v8 ignore next */
                (x = g), /* v8 ignore next */ /* v8 ignore next */
                x.test(v) && /* v8 ignore next */ /* v8 ignore next */
                  ((c = v), /* v8 ignore next */ /* v8 ignore next */
                  (x = k), /* v8 ignore next */ /* v8 ignore next */
                  (O = _), /* v8 ignore next */ /* v8 ignore next */
                  (M = B), /* v8 ignore next */ /* v8 ignore next */
                  x.test(c) /* v8 ignore next */ /* v8 ignore next */
                    ? (c = c + 'e') /* v8 ignore next */ /* v8 ignore next */
                    : O.test(c) /* v8 ignore next */ /* v8 ignore next */
                      ? ((h = w), (c = c.replace(h, ''))) /* v8 ignore next */ /* v8 ignore next */
                      : M.test(c) && (c = c + 'e'))); /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
            if (((h = A), h.test(c))) { /* v8 ignore next */ /* v8 ignore next */
              var E = h.exec(c); /* v8 ignore next */ /* v8 ignore next */
              ((v = E[1]), (c = v + 'i')); /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
            if (((h = j), h.test(c))) { /* v8 ignore next */ /* v8 ignore next */
              var E = h.exec(c); /* v8 ignore next */ /* v8 ignore next */
              ((v = E[1]), (P = E[2]), (h = y), h.test(v) && (c = v + e[P])); /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
            if (((h = $), h.test(c))) { /* v8 ignore next */ /* v8 ignore next */
              var E = h.exec(c); /* v8 ignore next */ /* v8 ignore next */
              ((v = E[1]), (P = E[2]), (h = y), h.test(v) && (c = v + n[P])); /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
            if (((h = V), (x = q), h.test(c))) { /* v8 ignore next */ /* v8 ignore next */
              var E = h.exec(c); /* v8 ignore next */ /* v8 ignore next */
              ((v = E[1]), (h = p), h.test(v) && (c = v)); /* v8 ignore next */ /* v8 ignore next */
            } else if (x.test(c)) { /* v8 ignore next */ /* v8 ignore next */
              var E = x.exec(c); /* v8 ignore next */ /* v8 ignore next */
              ((v = E[1] + E[2]), (x = p), x.test(v) && (c = v)); /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
            if (((h = C), h.test(c))) { /* v8 ignore next */ /* v8 ignore next */
              var E = h.exec(c); /* v8 ignore next */ /* v8 ignore next */
              ((v = E[1]), /* v8 ignore next */ /* v8 ignore next */
                (h = p), /* v8 ignore next */ /* v8 ignore next */
                (x = b), /* v8 ignore next */ /* v8 ignore next */
                (O = W), /* v8 ignore next */ /* v8 ignore next */
                (h.test(v) || (x.test(v) && !O.test(v))) && (c = v)); /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
            return ( /* v8 ignore next */ /* v8 ignore next */
              (h = z), /* v8 ignore next */ /* v8 ignore next */
              (x = p), /* v8 ignore next */ /* v8 ignore next */
              h.test(c) && x.test(c) && ((h = w), (c = c.replace(h, ''))), /* v8 ignore next */ /* v8 ignore next */
              T == 'y' && (c = T.toLowerCase() + c.substr(1)), /* v8 ignore next */ /* v8 ignore next */
              c /* v8 ignore next */ /* v8 ignore next */
            ); /* v8 ignore next */ /* v8 ignore next */
          }; /* v8 ignore next */ /* v8 ignore next */
        return function (R) { /* v8 ignore next */ /* v8 ignore next */
          return R.update(N); /* v8 ignore next */ /* v8 ignore next */
        }; /* v8 ignore next */ /* v8 ignore next */
      })()), /* v8 ignore next */ /* v8 ignore next */
        t.Pipeline.registerFunction(t.stemmer, 'stemmer')); /* v8 ignore next */ /* v8 ignore next */
      ((t.generateStopWordFilter = function (e) { /* v8 ignore next */ /* v8 ignore next */
        var n = e.reduce(function (r, i) { /* v8 ignore next */ /* v8 ignore next */
          return ((r[i] = i), r); /* v8 ignore next */ /* v8 ignore next */
        }, {}); /* v8 ignore next */ /* v8 ignore next */
        return function (r) { /* v8 ignore next */ /* v8 ignore next */
          if (r && n[r.toString()] !== r.toString()) return r; /* v8 ignore next */ /* v8 ignore next */
        }; /* v8 ignore next */ /* v8 ignore next */
      }), /* v8 ignore next */ /* v8 ignore next */
        (t.stopWordFilter = t.generateStopWordFilter([ /* v8 ignore next */ /* v8 ignore next */
          'a', /* v8 ignore next */ /* v8 ignore next */
          'able', /* v8 ignore next */ /* v8 ignore next */
          'about', /* v8 ignore next */ /* v8 ignore next */
          'across', /* v8 ignore next */ /* v8 ignore next */
          'after', /* v8 ignore next */ /* v8 ignore next */
          'all', /* v8 ignore next */ /* v8 ignore next */
          'almost', /* v8 ignore next */ /* v8 ignore next */
          'also', /* v8 ignore next */ /* v8 ignore next */
          'am', /* v8 ignore next */ /* v8 ignore next */
          'among', /* v8 ignore next */ /* v8 ignore next */
          'an', /* v8 ignore next */ /* v8 ignore next */
          'and', /* v8 ignore next */ /* v8 ignore next */
          'any', /* v8 ignore next */ /* v8 ignore next */
          'are', /* v8 ignore next */ /* v8 ignore next */
          'as', /* v8 ignore next */ /* v8 ignore next */
          'at', /* v8 ignore next */ /* v8 ignore next */
          'be', /* v8 ignore next */ /* v8 ignore next */
          'because', /* v8 ignore next */ /* v8 ignore next */
          'been', /* v8 ignore next */ /* v8 ignore next */
          'but', /* v8 ignore next */ /* v8 ignore next */
          'by', /* v8 ignore next */ /* v8 ignore next */
          'can', /* v8 ignore next */ /* v8 ignore next */
          'cannot', /* v8 ignore next */ /* v8 ignore next */
          'could', /* v8 ignore next */ /* v8 ignore next */
          'dear', /* v8 ignore next */ /* v8 ignore next */
          'did', /* v8 ignore next */ /* v8 ignore next */
          'do', /* v8 ignore next */ /* v8 ignore next */
          'does', /* v8 ignore next */ /* v8 ignore next */
          'either', /* v8 ignore next */ /* v8 ignore next */
          'else', /* v8 ignore next */ /* v8 ignore next */
          'ever', /* v8 ignore next */ /* v8 ignore next */
          'every', /* v8 ignore next */ /* v8 ignore next */
          'for', /* v8 ignore next */ /* v8 ignore next */
          'from', /* v8 ignore next */ /* v8 ignore next */
          'get', /* v8 ignore next */ /* v8 ignore next */
          'got', /* v8 ignore next */ /* v8 ignore next */
          'had', /* v8 ignore next */ /* v8 ignore next */
          'has', /* v8 ignore next */ /* v8 ignore next */
          'have', /* v8 ignore next */ /* v8 ignore next */
          'he', /* v8 ignore next */ /* v8 ignore next */
          'her', /* v8 ignore next */ /* v8 ignore next */
          'hers', /* v8 ignore next */ /* v8 ignore next */
          'him', /* v8 ignore next */ /* v8 ignore next */
          'his', /* v8 ignore next */ /* v8 ignore next */
          'how', /* v8 ignore next */ /* v8 ignore next */
          'however', /* v8 ignore next */ /* v8 ignore next */
          'i', /* v8 ignore next */ /* v8 ignore next */
          'if', /* v8 ignore next */ /* v8 ignore next */
          'in', /* v8 ignore next */ /* v8 ignore next */
          'into', /* v8 ignore next */ /* v8 ignore next */
          'is', /* v8 ignore next */ /* v8 ignore next */
          'it', /* v8 ignore next */ /* v8 ignore next */
          'its', /* v8 ignore next */ /* v8 ignore next */
          'just', /* v8 ignore next */ /* v8 ignore next */
          'least', /* v8 ignore next */ /* v8 ignore next */
          'let', /* v8 ignore next */ /* v8 ignore next */
          'like', /* v8 ignore next */ /* v8 ignore next */
          'likely', /* v8 ignore next */ /* v8 ignore next */
          'may', /* v8 ignore next */ /* v8 ignore next */
          'me', /* v8 ignore next */ /* v8 ignore next */
          'might', /* v8 ignore next */ /* v8 ignore next */
          'most', /* v8 ignore next */ /* v8 ignore next */
          'must', /* v8 ignore next */ /* v8 ignore next */
          'my', /* v8 ignore next */ /* v8 ignore next */
          'neither', /* v8 ignore next */ /* v8 ignore next */
          'no', /* v8 ignore next */ /* v8 ignore next */
          'nor', /* v8 ignore next */ /* v8 ignore next */
          'not', /* v8 ignore next */ /* v8 ignore next */
          'of', /* v8 ignore next */ /* v8 ignore next */
          'off', /* v8 ignore next */ /* v8 ignore next */
          'often', /* v8 ignore next */ /* v8 ignore next */
          'on', /* v8 ignore next */ /* v8 ignore next */
          'only', /* v8 ignore next */ /* v8 ignore next */
          'or', /* v8 ignore next */ /* v8 ignore next */
          'other', /* v8 ignore next */ /* v8 ignore next */
          'our', /* v8 ignore next */ /* v8 ignore next */
          'own', /* v8 ignore next */ /* v8 ignore next */
          'rather', /* v8 ignore next */ /* v8 ignore next */
          'said', /* v8 ignore next */ /* v8 ignore next */
          'say', /* v8 ignore next */ /* v8 ignore next */
          'says', /* v8 ignore next */ /* v8 ignore next */
          'she', /* v8 ignore next */ /* v8 ignore next */
          'should', /* v8 ignore next */ /* v8 ignore next */
          'since', /* v8 ignore next */ /* v8 ignore next */
          'so', /* v8 ignore next */ /* v8 ignore next */
          'some', /* v8 ignore next */ /* v8 ignore next */
          'than', /* v8 ignore next */ /* v8 ignore next */
          'that', /* v8 ignore next */ /* v8 ignore next */
          'the', /* v8 ignore next */ /* v8 ignore next */
          'their', /* v8 ignore next */ /* v8 ignore next */
          'them', /* v8 ignore next */ /* v8 ignore next */
          'then', /* v8 ignore next */ /* v8 ignore next */
          'there', /* v8 ignore next */ /* v8 ignore next */
          'these', /* v8 ignore next */ /* v8 ignore next */
          'they', /* v8 ignore next */ /* v8 ignore next */
          'this', /* v8 ignore next */ /* v8 ignore next */
          'tis', /* v8 ignore next */ /* v8 ignore next */
          'to', /* v8 ignore next */ /* v8 ignore next */
          'too', /* v8 ignore next */ /* v8 ignore next */
          'twas', /* v8 ignore next */ /* v8 ignore next */
          'us', /* v8 ignore next */ /* v8 ignore next */
          'wants', /* v8 ignore next */ /* v8 ignore next */
          'was', /* v8 ignore next */ /* v8 ignore next */
          'we', /* v8 ignore next */ /* v8 ignore next */
          'were', /* v8 ignore next */ /* v8 ignore next */
          'what', /* v8 ignore next */ /* v8 ignore next */
          'when', /* v8 ignore next */ /* v8 ignore next */
          'where', /* v8 ignore next */ /* v8 ignore next */
          'which', /* v8 ignore next */ /* v8 ignore next */
          'while', /* v8 ignore next */ /* v8 ignore next */
          'who', /* v8 ignore next */ /* v8 ignore next */
          'whom', /* v8 ignore next */ /* v8 ignore next */
          'why', /* v8 ignore next */ /* v8 ignore next */
          'will', /* v8 ignore next */ /* v8 ignore next */
          'with', /* v8 ignore next */ /* v8 ignore next */
          'would', /* v8 ignore next */ /* v8 ignore next */
          'yet', /* v8 ignore next */ /* v8 ignore next */
          'you', /* v8 ignore next */ /* v8 ignore next */
          'your' /* v8 ignore next */ /* v8 ignore next */
        ])), /* v8 ignore next */ /* v8 ignore next */
        t.Pipeline.registerFunction(t.stopWordFilter, 'stopWordFilter')); /* v8 ignore next */ /* v8 ignore next */
      ((t.trimmer = function (e) { /* v8 ignore next */ /* v8 ignore next */
        return e.update(function (n) { /* v8 ignore next */ /* v8 ignore next */
          return n.replace(/^\W+/, '').replace(/\W+$/, ''); /* v8 ignore next */ /* v8 ignore next */
        }); /* v8 ignore next */ /* v8 ignore next */
      }), /* v8 ignore next */ /* v8 ignore next */
        t.Pipeline.registerFunction(t.trimmer, 'trimmer')); /* v8 ignore next */ /* v8 ignore next */
      ((t.TokenSet = function () { /* v8 ignore next */ /* v8 ignore next */
        ((this.final = !1), /* v8 ignore next */ /* v8 ignore next */
          (this.edges = {}), /* v8 ignore next */ /* v8 ignore next */
          (this.id = t.TokenSet._nextId), /* v8 ignore next */ /* v8 ignore next */
          (t.TokenSet._nextId += 1)); /* v8 ignore next */ /* v8 ignore next */
      }), /* v8 ignore next */ /* v8 ignore next */
        (t.TokenSet._nextId = 1), /* v8 ignore next */ /* v8 ignore next */
        (t.TokenSet.fromArray = function (e) { /* v8 ignore next */ /* v8 ignore next */
          for (var n = new t.TokenSet.Builder(), r = 0, i = e.length; r < i; r++) n.insert(e[r]); /* v8 ignore next */ /* v8 ignore next */
          return (n.finish(), n.root); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.TokenSet.fromClause = function (e) { /* v8 ignore next */ /* v8 ignore next */
          return 'editDistance' in e /* v8 ignore next */ /* v8 ignore next */
            ? t.TokenSet.fromFuzzyString(e.term, e.editDistance) /* v8 ignore next */ /* v8 ignore next */
            : t.TokenSet.fromString(e.term); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.TokenSet.fromFuzzyString = function (e, n) { /* v8 ignore next */ /* v8 ignore next */
          for (var r = new t.TokenSet(), i = [{ node: r, editsRemaining: n, str: e }]; i.length; ) { /* v8 ignore next */ /* v8 ignore next */
            var s = i.pop(); /* v8 ignore next */ /* v8 ignore next */
            if (s.str.length > 0) { /* v8 ignore next */ /* v8 ignore next */
              var o = s.str.charAt(0), /* v8 ignore next */ /* v8 ignore next */
                a; /* v8 ignore next */ /* v8 ignore next */
              (o in s.node.edges /* v8 ignore next */ /* v8 ignore next */
                ? (a = s.node.edges[o]) /* v8 ignore next */ /* v8 ignore next */
                : ((a = new t.TokenSet()), (s.node.edges[o] = a)), /* v8 ignore next */ /* v8 ignore next */
                s.str.length == 1 && (a.final = !0), /* v8 ignore next */ /* v8 ignore next */
                i.push({ node: a, editsRemaining: s.editsRemaining, str: s.str.slice(1) })); /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
            if (s.editsRemaining != 0) { /* v8 ignore next */ /* v8 ignore next */
              if ('*' in s.node.edges) var l = s.node.edges['*']; /* v8 ignore next */ /* v8 ignore next */
              else { /* v8 ignore next */ /* v8 ignore next */
                var l = new t.TokenSet(); /* v8 ignore next */ /* v8 ignore next */
                s.node.edges['*'] = l; /* v8 ignore next */ /* v8 ignore next */
              } /* v8 ignore next */ /* v8 ignore next */
              if ( /* v8 ignore next */ /* v8 ignore next */
                (s.str.length == 0 && (l.final = !0), /* v8 ignore next */ /* v8 ignore next */
                i.push({ node: l, editsRemaining: s.editsRemaining - 1, str: s.str }), /* v8 ignore next */ /* v8 ignore next */
                s.str.length > 1 && /* v8 ignore next */ /* v8 ignore next */
                  i.push({ /* v8 ignore next */ /* v8 ignore next */
                    node: s.node, /* v8 ignore next */ /* v8 ignore next */
                    editsRemaining: s.editsRemaining - 1, /* v8 ignore next */ /* v8 ignore next */
                    str: s.str.slice(1) /* v8 ignore next */ /* v8 ignore next */
                  }), /* v8 ignore next */ /* v8 ignore next */
                s.str.length == 1 && (s.node.final = !0), /* v8 ignore next */ /* v8 ignore next */
                s.str.length >= 1) /* v8 ignore next */ /* v8 ignore next */
              ) { /* v8 ignore next */ /* v8 ignore next */
                if ('*' in s.node.edges) var u = s.node.edges['*']; /* v8 ignore next */ /* v8 ignore next */
                else { /* v8 ignore next */ /* v8 ignore next */
                  var u = new t.TokenSet(); /* v8 ignore next */ /* v8 ignore next */
                  s.node.edges['*'] = u; /* v8 ignore next */ /* v8 ignore next */
                } /* v8 ignore next */ /* v8 ignore next */
                (s.str.length == 1 && (u.final = !0), /* v8 ignore next */ /* v8 ignore next */
                  i.push({ node: u, editsRemaining: s.editsRemaining - 1, str: s.str.slice(1) })); /* v8 ignore next */ /* v8 ignore next */
              } /* v8 ignore next */ /* v8 ignore next */
              if (s.str.length > 1) { /* v8 ignore next */ /* v8 ignore next */
                var d = s.str.charAt(0), /* v8 ignore next */ /* v8 ignore next */
                  y = s.str.charAt(1), /* v8 ignore next */ /* v8 ignore next */
                  p; /* v8 ignore next */ /* v8 ignore next */
                (y in s.node.edges /* v8 ignore next */ /* v8 ignore next */
                  ? (p = s.node.edges[y]) /* v8 ignore next */ /* v8 ignore next */
                  : ((p = new t.TokenSet()), (s.node.edges[y] = p)), /* v8 ignore next */ /* v8 ignore next */
                  s.str.length == 1 && (p.final = !0), /* v8 ignore next */ /* v8 ignore next */
                  i.push({ /* v8 ignore next */ /* v8 ignore next */
                    node: p, /* v8 ignore next */ /* v8 ignore next */
                    editsRemaining: s.editsRemaining - 1, /* v8 ignore next */ /* v8 ignore next */
                    str: d + s.str.slice(2) /* v8 ignore next */ /* v8 ignore next */
                  })); /* v8 ignore next */ /* v8 ignore next */
              } /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
          return r; /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.TokenSet.fromString = function (e) { /* v8 ignore next */ /* v8 ignore next */
          for (var n = new t.TokenSet(), r = n, i = 0, s = e.length; i < s; i++) { /* v8 ignore next */ /* v8 ignore next */
            var o = e[i], /* v8 ignore next */ /* v8 ignore next */
              a = i == s - 1; /* v8 ignore next */ /* v8 ignore next */
            if (o == '*') ((n.edges[o] = n), (n.final = a)); /* v8 ignore next */ /* v8 ignore next */
            else { /* v8 ignore next */ /* v8 ignore next */
              var l = new t.TokenSet(); /* v8 ignore next */ /* v8 ignore next */
              ((l.final = a), (n.edges[o] = l), (n = l)); /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
          return r; /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.TokenSet.prototype.toArray = function () { /* v8 ignore next */ /* v8 ignore next */
          for (var e = [], n = [{ prefix: '', node: this }]; n.length; ) { /* v8 ignore next */ /* v8 ignore next */
            var r = n.pop(), /* v8 ignore next */ /* v8 ignore next */
              i = Object.keys(r.node.edges), /* v8 ignore next */ /* v8 ignore next */
              s = i.length; /* v8 ignore next */ /* v8 ignore next */
            r.node.final && (r.prefix.charAt(0), e.push(r.prefix)); /* v8 ignore next */ /* v8 ignore next */
            for (var o = 0; o < s; o++) { /* v8 ignore next */ /* v8 ignore next */
              var a = i[o]; /* v8 ignore next */ /* v8 ignore next */
              n.push({ prefix: r.prefix.concat(a), node: r.node.edges[a] }); /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
          return e; /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.TokenSet.prototype.toString = function () { /* v8 ignore next */ /* v8 ignore next */
          if (this._str) return this._str; /* v8 ignore next */ /* v8 ignore next */
          for ( /* v8 ignore next */ /* v8 ignore next */
            var e = this.final ? '1' : '0', n = Object.keys(this.edges).sort(), r = n.length, i = 0; /* v8 ignore next */ /* v8 ignore next */
            i < r; /* v8 ignore next */ /* v8 ignore next */
            i++ /* v8 ignore next */ /* v8 ignore next */
          ) { /* v8 ignore next */ /* v8 ignore next */
            var s = n[i], /* v8 ignore next */ /* v8 ignore next */
              o = this.edges[s]; /* v8 ignore next */ /* v8 ignore next */
            e = e + s + o.id; /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
          return e; /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.TokenSet.prototype.intersect = function (e) { /* v8 ignore next */ /* v8 ignore next */
          for ( /* v8 ignore next */ /* v8 ignore next */
            var n = new t.TokenSet(), r = void 0, i = [{ qNode: e, output: n, node: this }]; /* v8 ignore next */ /* v8 ignore next */
            i.length; /* v8 ignore next */ /* v8 ignore next */
          ) { /* v8 ignore next */ /* v8 ignore next */
            r = i.pop(); /* v8 ignore next */ /* v8 ignore next */
            for ( /* v8 ignore next */ /* v8 ignore next */
              var s = Object.keys(r.qNode.edges), /* v8 ignore next */ /* v8 ignore next */
                o = s.length, /* v8 ignore next */ /* v8 ignore next */
                a = Object.keys(r.node.edges), /* v8 ignore next */ /* v8 ignore next */
                l = a.length, /* v8 ignore next */ /* v8 ignore next */
                u = 0; /* v8 ignore next */ /* v8 ignore next */
              u < o; /* v8 ignore next */ /* v8 ignore next */
              u++ /* v8 ignore next */ /* v8 ignore next */
            ) /* v8 ignore next */ /* v8 ignore next */
              for (var d = s[u], y = 0; y < l; y++) { /* v8 ignore next */ /* v8 ignore next */
                var p = a[y]; /* v8 ignore next */ /* v8 ignore next */
                if (p == d || d == '*') { /* v8 ignore next */ /* v8 ignore next */
                  var b = r.node.edges[p], /* v8 ignore next */ /* v8 ignore next */
                    g = r.qNode.edges[d], /* v8 ignore next */ /* v8 ignore next */
                    L = b.final && g.final, /* v8 ignore next */ /* v8 ignore next */
                    f = void 0; /* v8 ignore next */ /* v8 ignore next */
                  (p in r.output.edges /* v8 ignore next */ /* v8 ignore next */
                    ? ((f = r.output.edges[p]), (f.final = f.final || L)) /* v8 ignore next */ /* v8 ignore next */
                    : ((f = new t.TokenSet()), (f.final = L), (r.output.edges[p] = f)), /* v8 ignore next */ /* v8 ignore next */
                    i.push({ qNode: g, output: f, node: b })); /* v8 ignore next */ /* v8 ignore next */
                } /* v8 ignore next */ /* v8 ignore next */
              } /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
          return n; /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.TokenSet.Builder = function () { /* v8 ignore next */ /* v8 ignore next */
          ((this.previousWord = ''), /* v8 ignore next */ /* v8 ignore next */
            (this.root = new t.TokenSet()), /* v8 ignore next */ /* v8 ignore next */
            (this.uncheckedNodes = []), /* v8 ignore next */ /* v8 ignore next */
            (this.minimizedNodes = {})); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.TokenSet.Builder.prototype.insert = function (e) { /* v8 ignore next */ /* v8 ignore next */
          var n, /* v8 ignore next */ /* v8 ignore next */
            r = 0; /* v8 ignore next */ /* v8 ignore next */
          if (e < this.previousWord) throw new Error('Out of order word insertion'); /* v8 ignore next */ /* v8 ignore next */
          for ( /* v8 ignore next */ /* v8 ignore next */
            var i = 0; /* v8 ignore next */ /* v8 ignore next */
            i < e.length && i < this.previousWord.length && e[i] == this.previousWord[i]; /* v8 ignore next */ /* v8 ignore next */
            i++ /* v8 ignore next */ /* v8 ignore next */
          ) /* v8 ignore next */ /* v8 ignore next */
            r++; /* v8 ignore next */ /* v8 ignore next */
          (this.minimize(r), /* v8 ignore next */ /* v8 ignore next */
            this.uncheckedNodes.length == 0 /* v8 ignore next */ /* v8 ignore next */
              ? (n = this.root) /* v8 ignore next */ /* v8 ignore next */
              : (n = this.uncheckedNodes[this.uncheckedNodes.length - 1].child)); /* v8 ignore next */ /* v8 ignore next */
          for (var i = r; i < e.length; i++) { /* v8 ignore next */ /* v8 ignore next */
            var s = new t.TokenSet(), /* v8 ignore next */ /* v8 ignore next */
              o = e[i]; /* v8 ignore next */ /* v8 ignore next */
            ((n.edges[o] = s), this.uncheckedNodes.push({ parent: n, char: o, child: s }), (n = s)); /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
          ((n.final = !0), (this.previousWord = e)); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.TokenSet.Builder.prototype.finish = function () { /* v8 ignore next */ /* v8 ignore next */
          this.minimize(0); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.TokenSet.Builder.prototype.minimize = function (e) { /* v8 ignore next */ /* v8 ignore next */
          for (var n = this.uncheckedNodes.length - 1; n >= e; n--) { /* v8 ignore next */ /* v8 ignore next */
            var r = this.uncheckedNodes[n], /* v8 ignore next */ /* v8 ignore next */
              i = r.child.toString(); /* v8 ignore next */ /* v8 ignore next */
            (i in this.minimizedNodes /* v8 ignore next */ /* v8 ignore next */
              ? (r.parent.edges[r.char] = this.minimizedNodes[i]) /* v8 ignore next */ /* v8 ignore next */
              : ((r.child._str = i), (this.minimizedNodes[i] = r.child)), /* v8 ignore next */ /* v8 ignore next */
              this.uncheckedNodes.pop()); /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
        })); /* v8 ignore next */ /* v8 ignore next */
      ((t.Index = function (e) { /* v8 ignore next */ /* v8 ignore next */
        ((this.invertedIndex = e.invertedIndex), /* v8 ignore next */ /* v8 ignore next */
          (this.fieldVectors = e.fieldVectors), /* v8 ignore next */ /* v8 ignore next */
          (this.tokenSet = e.tokenSet), /* v8 ignore next */ /* v8 ignore next */
          (this.fields = e.fields), /* v8 ignore next */ /* v8 ignore next */
          (this.pipeline = e.pipeline)); /* v8 ignore next */ /* v8 ignore next */
      }), /* v8 ignore next */ /* v8 ignore next */
        (t.Index.prototype.search = function (e) { /* v8 ignore next */ /* v8 ignore next */
          return this.query(function (n) { /* v8 ignore next */ /* v8 ignore next */
            var r = new t.QueryParser(e, n); /* v8 ignore next */ /* v8 ignore next */
            r.parse(); /* v8 ignore next */ /* v8 ignore next */
          }); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.Index.prototype.query = function (e) { /* v8 ignore next */ /* v8 ignore next */
          for ( /* v8 ignore next */ /* v8 ignore next */
            var n = new t.Query(this.fields), /* v8 ignore next */ /* v8 ignore next */
              r = Object.create(null), /* v8 ignore next */ /* v8 ignore next */
              i = Object.create(null), /* v8 ignore next */ /* v8 ignore next */
              s = Object.create(null), /* v8 ignore next */ /* v8 ignore next */
              o = Object.create(null), /* v8 ignore next */ /* v8 ignore next */
              a = Object.create(null), /* v8 ignore next */ /* v8 ignore next */
              l = 0; /* v8 ignore next */ /* v8 ignore next */
            l < this.fields.length; /* v8 ignore next */ /* v8 ignore next */
            l++ /* v8 ignore next */ /* v8 ignore next */
          ) /* v8 ignore next */ /* v8 ignore next */
            i[this.fields[l]] = new t.Vector(); /* v8 ignore next */ /* v8 ignore next */
          e.call(n, n); /* v8 ignore next */ /* v8 ignore next */
          for (var l = 0; l < n.clauses.length; l++) { /* v8 ignore next */ /* v8 ignore next */
            var u = n.clauses[l], /* v8 ignore next */ /* v8 ignore next */
              d = null, /* v8 ignore next */ /* v8 ignore next */
              y = t.Set.empty; /* v8 ignore next */ /* v8 ignore next */
            u.usePipeline /* v8 ignore next */ /* v8 ignore next */
              ? (d = this.pipeline.runString(u.term, { fields: u.fields })) /* v8 ignore next */ /* v8 ignore next */
              : (d = [u.term]); /* v8 ignore next */ /* v8 ignore next */
            for (var p = 0; p < d.length; p++) { /* v8 ignore next */ /* v8 ignore next */
              var b = d[p]; /* v8 ignore next */ /* v8 ignore next */
              u.term = b; /* v8 ignore next */ /* v8 ignore next */
              var g = t.TokenSet.fromClause(u), /* v8 ignore next */ /* v8 ignore next */
                L = this.tokenSet.intersect(g).toArray(); /* v8 ignore next */ /* v8 ignore next */
              if (L.length === 0 && u.presence === t.Query.presence.REQUIRED) { /* v8 ignore next */ /* v8 ignore next */
                for (var f = 0; f < u.fields.length; f++) { /* v8 ignore next */ /* v8 ignore next */
                  var m = u.fields[f]; /* v8 ignore next */ /* v8 ignore next */
                  o[m] = t.Set.empty; /* v8 ignore next */ /* v8 ignore next */
                } /* v8 ignore next */ /* v8 ignore next */
                break; /* v8 ignore next */ /* v8 ignore next */
              } /* v8 ignore next */ /* v8 ignore next */
              for (var S = 0; S < L.length; S++) /* v8 ignore next */ /* v8 ignore next */
                for ( /* v8 ignore next */ /* v8 ignore next */
                  var w = L[S], k = this.invertedIndex[w], _ = k._index, f = 0; /* v8 ignore next */ /* v8 ignore next */
                  f < u.fields.length; /* v8 ignore next */ /* v8 ignore next */
                  f++ /* v8 ignore next */ /* v8 ignore next */
                ) { /* v8 ignore next */ /* v8 ignore next */
                  var m = u.fields[f], /* v8 ignore next */ /* v8 ignore next */
                    B = k[m], /* v8 ignore next */ /* v8 ignore next */
                    A = Object.keys(B), /* v8 ignore next */ /* v8 ignore next */
                    j = w + '/' + m, /* v8 ignore next */ /* v8 ignore next */
                    $ = new t.Set(A); /* v8 ignore next */ /* v8 ignore next */
                  if ( /* v8 ignore next */ /* v8 ignore next */
                    (u.presence == t.Query.presence.REQUIRED && /* v8 ignore next */ /* v8 ignore next */
                      ((y = y.union($)), o[m] === void 0 && (o[m] = t.Set.complete)), /* v8 ignore next */ /* v8 ignore next */
                    u.presence == t.Query.presence.PROHIBITED) /* v8 ignore next */ /* v8 ignore next */
                  ) { /* v8 ignore next */ /* v8 ignore next */
                    (a[m] === void 0 && (a[m] = t.Set.empty), (a[m] = a[m].union($))); /* v8 ignore next */ /* v8 ignore next */
                    continue; /* v8 ignore next */ /* v8 ignore next */
                  } /* v8 ignore next */ /* v8 ignore next */
                  if ( /* v8 ignore next */ /* v8 ignore next */
                    (i[m].upsert(_, u.boost, function (Qe, Ie) { /* v8 ignore next */ /* v8 ignore next */
                      return Qe + Ie; /* v8 ignore next */ /* v8 ignore next */
                    }), /* v8 ignore next */ /* v8 ignore next */
                    !s[j]) /* v8 ignore next */ /* v8 ignore next */
                  ) { /* v8 ignore next */ /* v8 ignore next */
                    for (var V = 0; V < A.length; V++) { /* v8 ignore next */ /* v8 ignore next */
                      var q = A[V], /* v8 ignore next */ /* v8 ignore next */
                        C = new t.FieldRef(q, m), /* v8 ignore next */ /* v8 ignore next */
                        z = B[q], /* v8 ignore next */ /* v8 ignore next */
                        W; /* v8 ignore next */ /* v8 ignore next */
                      (W = r[C]) === void 0 ? (r[C] = new t.MatchData(w, m, z)) : W.add(w, m, z); /* v8 ignore next */ /* v8 ignore next */
                    } /* v8 ignore next */ /* v8 ignore next */
                    s[j] = !0; /* v8 ignore next */ /* v8 ignore next */
                  } /* v8 ignore next */ /* v8 ignore next */
                } /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
            if (u.presence === t.Query.presence.REQUIRED) /* v8 ignore next */ /* v8 ignore next */
              for (var f = 0; f < u.fields.length; f++) { /* v8 ignore next */ /* v8 ignore next */
                var m = u.fields[f]; /* v8 ignore next */ /* v8 ignore next */
                o[m] = o[m].intersect(y); /* v8 ignore next */ /* v8 ignore next */
              } /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
          for (var N = t.Set.complete, R = t.Set.empty, l = 0; l < this.fields.length; l++) { /* v8 ignore next */ /* v8 ignore next */
            var m = this.fields[l]; /* v8 ignore next */ /* v8 ignore next */
            (o[m] && (N = N.intersect(o[m])), a[m] && (R = R.union(a[m]))); /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
          var c = Object.keys(r), /* v8 ignore next */ /* v8 ignore next */
            v = [], /* v8 ignore next */ /* v8 ignore next */
            P = Object.create(null); /* v8 ignore next */ /* v8 ignore next */
          if (n.isNegated()) { /* v8 ignore next */ /* v8 ignore next */
            c = Object.keys(this.fieldVectors); /* v8 ignore next */ /* v8 ignore next */
            for (var l = 0; l < c.length; l++) { /* v8 ignore next */ /* v8 ignore next */
              var C = c[l], /* v8 ignore next */ /* v8 ignore next */
                T = t.FieldRef.fromString(C); /* v8 ignore next */ /* v8 ignore next */
              r[C] = new t.MatchData(); /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
          for (var l = 0; l < c.length; l++) { /* v8 ignore next */ /* v8 ignore next */
            var T = t.FieldRef.fromString(c[l]), /* v8 ignore next */ /* v8 ignore next */
              h = T.docRef; /* v8 ignore next */ /* v8 ignore next */
            if (N.contains(h) && !R.contains(h)) { /* v8 ignore next */ /* v8 ignore next */
              var x = this.fieldVectors[T], /* v8 ignore next */ /* v8 ignore next */
                O = i[T.fieldName].similarity(x), /* v8 ignore next */ /* v8 ignore next */
                M; /* v8 ignore next */ /* v8 ignore next */
              if ((M = P[h]) !== void 0) ((M.score += O), M.matchData.combine(r[T])); /* v8 ignore next */ /* v8 ignore next */
              else { /* v8 ignore next */ /* v8 ignore next */
                var E = { ref: h, score: O, matchData: r[T] }; /* v8 ignore next */ /* v8 ignore next */
                ((P[h] = E), v.push(E)); /* v8 ignore next */ /* v8 ignore next */
              } /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
          return v.sort(function (Te, ke) { /* v8 ignore next */ /* v8 ignore next */
            return ke.score - Te.score; /* v8 ignore next */ /* v8 ignore next */
          }); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.Index.prototype.toJSON = function () { /* v8 ignore next */ /* v8 ignore next */
          var e = Object.keys(this.invertedIndex) /* v8 ignore next */ /* v8 ignore next */
              .sort() /* v8 ignore next */ /* v8 ignore next */
              .map(function (r) { /* v8 ignore next */ /* v8 ignore next */
                return [r, this.invertedIndex[r]]; /* v8 ignore next */ /* v8 ignore next */
              }, this), /* v8 ignore next */ /* v8 ignore next */
            n = Object.keys(this.fieldVectors).map(function (r) { /* v8 ignore next */ /* v8 ignore next */
              return [r, this.fieldVectors[r].toJSON()]; /* v8 ignore next */ /* v8 ignore next */
            }, this); /* v8 ignore next */ /* v8 ignore next */
          return { /* v8 ignore next */ /* v8 ignore next */
            version: t.version, /* v8 ignore next */ /* v8 ignore next */
            fields: this.fields, /* v8 ignore next */ /* v8 ignore next */
            fieldVectors: n, /* v8 ignore next */ /* v8 ignore next */
            invertedIndex: e, /* v8 ignore next */ /* v8 ignore next */
            pipeline: this.pipeline.toJSON() /* v8 ignore next */ /* v8 ignore next */
          }; /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.Index.load = function (e) { /* v8 ignore next */ /* v8 ignore next */
          var n = {}, /* v8 ignore next */ /* v8 ignore next */
            r = {}, /* v8 ignore next */ /* v8 ignore next */
            i = e.fieldVectors, /* v8 ignore next */ /* v8 ignore next */
            s = Object.create(null), /* v8 ignore next */ /* v8 ignore next */
            o = e.invertedIndex, /* v8 ignore next */ /* v8 ignore next */
            a = new t.TokenSet.Builder(), /* v8 ignore next */ /* v8 ignore next */
            l = t.Pipeline.load(e.pipeline); /* v8 ignore next */ /* v8 ignore next */
          e.version != t.version && /* v8 ignore next */ /* v8 ignore next */
            t.utils.warn( /* v8 ignore next */ /* v8 ignore next */
              "Version mismatch when loading serialised index. Current version of lunr '" + /* v8 ignore next */ /* v8 ignore next */
                t.version + /* v8 ignore next */ /* v8 ignore next */
                "' does not match serialized index '" + /* v8 ignore next */ /* v8 ignore next */
                e.version + /* v8 ignore next */ /* v8 ignore next */
                "'" /* v8 ignore next */ /* v8 ignore next */
            ); /* v8 ignore next */ /* v8 ignore next */
          for (var u = 0; u < i.length; u++) { /* v8 ignore next */ /* v8 ignore next */
            var d = i[u], /* v8 ignore next */ /* v8 ignore next */
              y = d[0], /* v8 ignore next */ /* v8 ignore next */
              p = d[1]; /* v8 ignore next */ /* v8 ignore next */
            r[y] = new t.Vector(p); /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
          for (var u = 0; u < o.length; u++) { /* v8 ignore next */ /* v8 ignore next */
            var d = o[u], /* v8 ignore next */ /* v8 ignore next */
              b = d[0], /* v8 ignore next */ /* v8 ignore next */
              g = d[1]; /* v8 ignore next */ /* v8 ignore next */
            (a.insert(b), (s[b] = g)); /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
          return ( /* v8 ignore next */ /* v8 ignore next */
            a.finish(), /* v8 ignore next */ /* v8 ignore next */
            (n.fields = e.fields), /* v8 ignore next */ /* v8 ignore next */
            (n.fieldVectors = r), /* v8 ignore next */ /* v8 ignore next */
            (n.invertedIndex = s), /* v8 ignore next */ /* v8 ignore next */
            (n.tokenSet = a.root), /* v8 ignore next */ /* v8 ignore next */
            (n.pipeline = l), /* v8 ignore next */ /* v8 ignore next */
            new t.Index(n) /* v8 ignore next */ /* v8 ignore next */
          ); /* v8 ignore next */ /* v8 ignore next */
        })); /* v8 ignore next */ /* v8 ignore next */
      ((t.Builder = function () { /* v8 ignore next */ /* v8 ignore next */
        ((this._ref = 'id'), /* v8 ignore next */ /* v8 ignore next */
          (this._fields = Object.create(null)), /* v8 ignore next */ /* v8 ignore next */
          (this._documents = Object.create(null)), /* v8 ignore next */ /* v8 ignore next */
          (this.invertedIndex = Object.create(null)), /* v8 ignore next */ /* v8 ignore next */
          (this.fieldTermFrequencies = {}), /* v8 ignore next */ /* v8 ignore next */
          (this.fieldLengths = {}), /* v8 ignore next */ /* v8 ignore next */
          (this.tokenizer = t.tokenizer), /* v8 ignore next */ /* v8 ignore next */
          (this.pipeline = new t.Pipeline()), /* v8 ignore next */ /* v8 ignore next */
          (this.searchPipeline = new t.Pipeline()), /* v8 ignore next */ /* v8 ignore next */
          (this.documentCount = 0), /* v8 ignore next */ /* v8 ignore next */
          (this._b = 0.75), /* v8 ignore next */ /* v8 ignore next */
          (this._k1 = 1.2), /* v8 ignore next */ /* v8 ignore next */
          (this.termIndex = 0), /* v8 ignore next */ /* v8 ignore next */
          (this.metadataWhitelist = [])); /* v8 ignore next */ /* v8 ignore next */
      }), /* v8 ignore next */ /* v8 ignore next */
        (t.Builder.prototype.ref = function (e) { /* v8 ignore next */ /* v8 ignore next */
          this._ref = e; /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.Builder.prototype.field = function (e, n) { /* v8 ignore next */ /* v8 ignore next */
          if (/\//.test(e)) /* v8 ignore next */ /* v8 ignore next */
            throw new RangeError("Field '" + e + "' contains illegal character '/'"); /* v8 ignore next */ /* v8 ignore next */
          this._fields[e] = n || {}; /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.Builder.prototype.b = function (e) { /* v8 ignore next */ /* v8 ignore next */
          e < 0 ? (this._b = 0) : e > 1 ? (this._b = 1) : (this._b = e); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.Builder.prototype.k1 = function (e) { /* v8 ignore next */ /* v8 ignore next */
          this._k1 = e; /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.Builder.prototype.add = function (e, n) { /* v8 ignore next */ /* v8 ignore next */
          var r = e[this._ref], /* v8 ignore next */ /* v8 ignore next */
            i = Object.keys(this._fields); /* v8 ignore next */ /* v8 ignore next */
          ((this._documents[r] = n || {}), (this.documentCount += 1)); /* v8 ignore next */ /* v8 ignore next */
          for (var s = 0; s < i.length; s++) { /* v8 ignore next */ /* v8 ignore next */
            var o = i[s], /* v8 ignore next */ /* v8 ignore next */
              a = this._fields[o].extractor, /* v8 ignore next */ /* v8 ignore next */
              l = a ? a(e) : e[o], /* v8 ignore next */ /* v8 ignore next */
              u = this.tokenizer(l, { fields: [o] }), /* v8 ignore next */ /* v8 ignore next */
              d = this.pipeline.run(u), /* v8 ignore next */ /* v8 ignore next */
              y = new t.FieldRef(r, o), /* v8 ignore next */ /* v8 ignore next */
              p = Object.create(null); /* v8 ignore next */ /* v8 ignore next */
            ((this.fieldTermFrequencies[y] = p), /* v8 ignore next */ /* v8 ignore next */
              (this.fieldLengths[y] = 0), /* v8 ignore next */ /* v8 ignore next */
              (this.fieldLengths[y] += d.length)); /* v8 ignore next */ /* v8 ignore next */
            for (var b = 0; b < d.length; b++) { /* v8 ignore next */ /* v8 ignore next */
              var g = d[b]; /* v8 ignore next */ /* v8 ignore next */
              if ((p[g] == null && (p[g] = 0), (p[g] += 1), this.invertedIndex[g] == null)) { /* v8 ignore next */ /* v8 ignore next */
                var L = Object.create(null); /* v8 ignore next */ /* v8 ignore next */
                ((L._index = this.termIndex), (this.termIndex += 1)); /* v8 ignore next */ /* v8 ignore next */
                for (var f = 0; f < i.length; f++) L[i[f]] = Object.create(null); /* v8 ignore next */ /* v8 ignore next */
                this.invertedIndex[g] = L; /* v8 ignore next */ /* v8 ignore next */
              } /* v8 ignore next */ /* v8 ignore next */
              this.invertedIndex[g][o][r] == null && /* v8 ignore next */ /* v8 ignore next */
                (this.invertedIndex[g][o][r] = Object.create(null)); /* v8 ignore next */ /* v8 ignore next */
              for (var m = 0; m < this.metadataWhitelist.length; m++) { /* v8 ignore next */ /* v8 ignore next */
                var S = this.metadataWhitelist[m], /* v8 ignore next */ /* v8 ignore next */
                  w = g.metadata[S]; /* v8 ignore next */ /* v8 ignore next */
                (this.invertedIndex[g][o][r][S] == null && (this.invertedIndex[g][o][r][S] = []), /* v8 ignore next */ /* v8 ignore next */
                  this.invertedIndex[g][o][r][S].push(w)); /* v8 ignore next */ /* v8 ignore next */
              } /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.Builder.prototype.calculateAverageFieldLengths = function () { /* v8 ignore next */ /* v8 ignore next */
          for ( /* v8 ignore next */ /* v8 ignore next */
            var e = Object.keys(this.fieldLengths), n = e.length, r = {}, i = {}, s = 0; /* v8 ignore next */ /* v8 ignore next */
            s < n; /* v8 ignore next */ /* v8 ignore next */
            s++ /* v8 ignore next */ /* v8 ignore next */
          ) { /* v8 ignore next */ /* v8 ignore next */
            var o = t.FieldRef.fromString(e[s]), /* v8 ignore next */ /* v8 ignore next */
              a = o.fieldName; /* v8 ignore next */ /* v8 ignore next */
            (i[a] || (i[a] = 0), (i[a] += 1), r[a] || (r[a] = 0), (r[a] += this.fieldLengths[o])); /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
          for (var l = Object.keys(this._fields), s = 0; s < l.length; s++) { /* v8 ignore next */ /* v8 ignore next */
            var u = l[s]; /* v8 ignore next */ /* v8 ignore next */
            r[u] = r[u] / i[u]; /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
          this.averageFieldLength = r; /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.Builder.prototype.createFieldVectors = function () { /* v8 ignore next */ /* v8 ignore next */
          for ( /* v8 ignore next */ /* v8 ignore next */
            var e = {}, /* v8 ignore next */ /* v8 ignore next */
              n = Object.keys(this.fieldTermFrequencies), /* v8 ignore next */ /* v8 ignore next */
              r = n.length, /* v8 ignore next */ /* v8 ignore next */
              i = Object.create(null), /* v8 ignore next */ /* v8 ignore next */
              s = 0; /* v8 ignore next */ /* v8 ignore next */
            s < r; /* v8 ignore next */ /* v8 ignore next */
            s++ /* v8 ignore next */ /* v8 ignore next */
          ) { /* v8 ignore next */ /* v8 ignore next */
            for ( /* v8 ignore next */ /* v8 ignore next */
              var o = t.FieldRef.fromString(n[s]), /* v8 ignore next */ /* v8 ignore next */
                a = o.fieldName, /* v8 ignore next */ /* v8 ignore next */
                l = this.fieldLengths[o], /* v8 ignore next */ /* v8 ignore next */
                u = new t.Vector(), /* v8 ignore next */ /* v8 ignore next */
                d = this.fieldTermFrequencies[o], /* v8 ignore next */ /* v8 ignore next */
                y = Object.keys(d), /* v8 ignore next */ /* v8 ignore next */
                p = y.length, /* v8 ignore next */ /* v8 ignore next */
                b = this._fields[a].boost || 1, /* v8 ignore next */ /* v8 ignore next */
                g = this._documents[o.docRef].boost || 1, /* v8 ignore next */ /* v8 ignore next */
                L = 0; /* v8 ignore next */ /* v8 ignore next */
              L < p; /* v8 ignore next */ /* v8 ignore next */
              L++ /* v8 ignore next */ /* v8 ignore next */
            ) { /* v8 ignore next */ /* v8 ignore next */
              var f = y[L], /* v8 ignore next */ /* v8 ignore next */
                m = d[f], /* v8 ignore next */ /* v8 ignore next */
                S = this.invertedIndex[f]._index, /* v8 ignore next */ /* v8 ignore next */
                w, /* v8 ignore next */ /* v8 ignore next */
                k, /* v8 ignore next */ /* v8 ignore next */
                _; /* v8 ignore next */ /* v8 ignore next */
              (i[f] === void 0 /* v8 ignore next */ /* v8 ignore next */
                ? ((w = t.idf(this.invertedIndex[f], this.documentCount)), (i[f] = w)) /* v8 ignore next */ /* v8 ignore next */
                : (w = i[f]), /* v8 ignore next */ /* v8 ignore next */
                (k = /* v8 ignore next */ /* v8 ignore next */
                  (w * ((this._k1 + 1) * m)) / /* v8 ignore next */ /* v8 ignore next */
                  (this._k1 * (1 - this._b + this._b * (l / this.averageFieldLength[a])) + m)), /* v8 ignore next */ /* v8 ignore next */
                (k *= b), /* v8 ignore next */ /* v8 ignore next */
                (k *= g), /* v8 ignore next */ /* v8 ignore next */
                (_ = Math.round(k * 1e3) / 1e3), /* v8 ignore next */ /* v8 ignore next */
                u.insert(S, _)); /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
            e[o] = u; /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
          this.fieldVectors = e; /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.Builder.prototype.createTokenSet = function () { /* v8 ignore next */ /* v8 ignore next */
          this.tokenSet = t.TokenSet.fromArray(Object.keys(this.invertedIndex).sort()); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.Builder.prototype.build = function () { /* v8 ignore next */ /* v8 ignore next */
          return ( /* v8 ignore next */ /* v8 ignore next */
            this.calculateAverageFieldLengths(), /* v8 ignore next */ /* v8 ignore next */
            this.createFieldVectors(), /* v8 ignore next */ /* v8 ignore next */
            this.createTokenSet(), /* v8 ignore next */ /* v8 ignore next */
            new t.Index({ /* v8 ignore next */ /* v8 ignore next */
              invertedIndex: this.invertedIndex, /* v8 ignore next */ /* v8 ignore next */
              fieldVectors: this.fieldVectors, /* v8 ignore next */ /* v8 ignore next */
              tokenSet: this.tokenSet, /* v8 ignore next */ /* v8 ignore next */
              fields: Object.keys(this._fields), /* v8 ignore next */ /* v8 ignore next */
              pipeline: this.searchPipeline /* v8 ignore next */ /* v8 ignore next */
            }) /* v8 ignore next */ /* v8 ignore next */
          ); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.Builder.prototype.use = function (e) { /* v8 ignore next */ /* v8 ignore next */
          var n = Array.prototype.slice.call(arguments, 1); /* v8 ignore next */ /* v8 ignore next */
          (n.unshift(this), e.apply(this, n)); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.MatchData = function (e, n, r) { /* v8 ignore next */ /* v8 ignore next */
          for (var i = Object.create(null), s = Object.keys(r || {}), o = 0; o < s.length; o++) { /* v8 ignore next */ /* v8 ignore next */
            var a = s[o]; /* v8 ignore next */ /* v8 ignore next */
            i[a] = r[a].slice(); /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
          ((this.metadata = Object.create(null)), /* v8 ignore next */ /* v8 ignore next */
            e !== void 0 && ((this.metadata[e] = Object.create(null)), (this.metadata[e][n] = i))); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.MatchData.prototype.combine = function (e) { /* v8 ignore next */ /* v8 ignore next */
          for (var n = Object.keys(e.metadata), r = 0; r < n.length; r++) { /* v8 ignore next */ /* v8 ignore next */
            var i = n[r], /* v8 ignore next */ /* v8 ignore next */
              s = Object.keys(e.metadata[i]); /* v8 ignore next */ /* v8 ignore next */
            this.metadata[i] == null && (this.metadata[i] = Object.create(null)); /* v8 ignore next */ /* v8 ignore next */
            for (var o = 0; o < s.length; o++) { /* v8 ignore next */ /* v8 ignore next */
              var a = s[o], /* v8 ignore next */ /* v8 ignore next */
                l = Object.keys(e.metadata[i][a]); /* v8 ignore next */ /* v8 ignore next */
              this.metadata[i][a] == null && (this.metadata[i][a] = Object.create(null)); /* v8 ignore next */ /* v8 ignore next */
              for (var u = 0; u < l.length; u++) { /* v8 ignore next */ /* v8 ignore next */
                var d = l[u]; /* v8 ignore next */ /* v8 ignore next */
                this.metadata[i][a][d] == null /* v8 ignore next */ /* v8 ignore next */
                  ? (this.metadata[i][a][d] = e.metadata[i][a][d]) /* v8 ignore next */ /* v8 ignore next */
                  : (this.metadata[i][a][d] = this.metadata[i][a][d].concat(e.metadata[i][a][d])); /* v8 ignore next */ /* v8 ignore next */
              } /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.MatchData.prototype.add = function (e, n, r) { /* v8 ignore next */ /* v8 ignore next */
          if (!(e in this.metadata)) { /* v8 ignore next */ /* v8 ignore next */
            ((this.metadata[e] = Object.create(null)), (this.metadata[e][n] = r)); /* v8 ignore next */ /* v8 ignore next */
            return; /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
          if (!(n in this.metadata[e])) { /* v8 ignore next */ /* v8 ignore next */
            this.metadata[e][n] = r; /* v8 ignore next */ /* v8 ignore next */
            return; /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
          for (var i = Object.keys(r), s = 0; s < i.length; s++) { /* v8 ignore next */ /* v8 ignore next */
            var o = i[s]; /* v8 ignore next */ /* v8 ignore next */
            o in this.metadata[e][n] /* v8 ignore next */ /* v8 ignore next */
              ? (this.metadata[e][n][o] = this.metadata[e][n][o].concat(r[o])) /* v8 ignore next */ /* v8 ignore next */
              : (this.metadata[e][n][o] = r[o]); /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.Query = function (e) { /* v8 ignore next */ /* v8 ignore next */
          ((this.clauses = []), (this.allFields = e)); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.Query.wildcard = new String('*')), /* v8 ignore next */ /* v8 ignore next */
        (t.Query.wildcard.NONE = 0), /* v8 ignore next */ /* v8 ignore next */
        (t.Query.wildcard.LEADING = 1), /* v8 ignore next */ /* v8 ignore next */
        (t.Query.wildcard.TRAILING = 2), /* v8 ignore next */ /* v8 ignore next */
        (t.Query.presence = { OPTIONAL: 1, REQUIRED: 2, PROHIBITED: 3 }), /* v8 ignore next */ /* v8 ignore next */
        (t.Query.prototype.clause = function (e) { /* v8 ignore next */ /* v8 ignore next */
          return ( /* v8 ignore next */ /* v8 ignore next */
            'fields' in e || (e.fields = this.allFields), /* v8 ignore next */ /* v8 ignore next */
            'boost' in e || (e.boost = 1), /* v8 ignore next */ /* v8 ignore next */
            'usePipeline' in e || (e.usePipeline = !0), /* v8 ignore next */ /* v8 ignore next */
            'wildcard' in e || (e.wildcard = t.Query.wildcard.NONE), /* v8 ignore next */ /* v8 ignore next */
            e.wildcard & t.Query.wildcard.LEADING && /* v8 ignore next */ /* v8 ignore next */
              e.term.charAt(0) != t.Query.wildcard && /* v8 ignore next */ /* v8 ignore next */
              (e.term = '*' + e.term), /* v8 ignore next */ /* v8 ignore next */
            e.wildcard & t.Query.wildcard.TRAILING && /* v8 ignore next */ /* v8 ignore next */
              e.term.slice(-1) != t.Query.wildcard && /* v8 ignore next */ /* v8 ignore next */
              (e.term = '' + e.term + '*'), /* v8 ignore next */ /* v8 ignore next */
            'presence' in e || (e.presence = t.Query.presence.OPTIONAL), /* v8 ignore next */ /* v8 ignore next */
            this.clauses.push(e), /* v8 ignore next */ /* v8 ignore next */
            this /* v8 ignore next */ /* v8 ignore next */
          ); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.Query.prototype.isNegated = function () { /* v8 ignore next */ /* v8 ignore next */
          for (var e = 0; e < this.clauses.length; e++) /* v8 ignore next */ /* v8 ignore next */
            if (this.clauses[e].presence != t.Query.presence.PROHIBITED) return !1; /* v8 ignore next */ /* v8 ignore next */
          return !0; /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.Query.prototype.term = function (e, n) { /* v8 ignore next */ /* v8 ignore next */
          if (Array.isArray(e)) /* v8 ignore next */ /* v8 ignore next */
            return ( /* v8 ignore next */ /* v8 ignore next */
              e.forEach(function (i) { /* v8 ignore next */ /* v8 ignore next */
                this.term(i, t.utils.clone(n)); /* v8 ignore next */ /* v8 ignore next */
              }, this), /* v8 ignore next */ /* v8 ignore next */
              this /* v8 ignore next */ /* v8 ignore next */
            ); /* v8 ignore next */ /* v8 ignore next */
          var r = n || {}; /* v8 ignore next */ /* v8 ignore next */
          return ((r.term = e.toString()), this.clause(r), this); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.QueryParseError = function (e, n, r) { /* v8 ignore next */ /* v8 ignore next */
          ((this.name = 'QueryParseError'), (this.message = e), (this.start = n), (this.end = r)); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.QueryParseError.prototype = new Error()), /* v8 ignore next */ /* v8 ignore next */
        (t.QueryLexer = function (e) { /* v8 ignore next */ /* v8 ignore next */
          ((this.lexemes = []), /* v8 ignore next */ /* v8 ignore next */
            (this.str = e), /* v8 ignore next */ /* v8 ignore next */
            (this.length = e.length), /* v8 ignore next */ /* v8 ignore next */
            (this.pos = 0), /* v8 ignore next */ /* v8 ignore next */
            (this.start = 0), /* v8 ignore next */ /* v8 ignore next */
            (this.escapeCharPositions = [])); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.QueryLexer.prototype.run = function () { /* v8 ignore next */ /* v8 ignore next */
          for (var e = t.QueryLexer.lexText; e; ) e = e(this); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.QueryLexer.prototype.sliceString = function () { /* v8 ignore next */ /* v8 ignore next */
          for ( /* v8 ignore next */ /* v8 ignore next */
            var e = [], n = this.start, r = this.pos, i = 0; /* v8 ignore next */ /* v8 ignore next */
            i < this.escapeCharPositions.length; /* v8 ignore next */ /* v8 ignore next */
            i++ /* v8 ignore next */ /* v8 ignore next */
          ) /* v8 ignore next */ /* v8 ignore next */
            ((r = this.escapeCharPositions[i]), e.push(this.str.slice(n, r)), (n = r + 1)); /* v8 ignore next */ /* v8 ignore next */
          return ( /* v8 ignore next */ /* v8 ignore next */
            e.push(this.str.slice(n, this.pos)), /* v8 ignore next */ /* v8 ignore next */
            (this.escapeCharPositions.length = 0), /* v8 ignore next */ /* v8 ignore next */
            e.join('') /* v8 ignore next */ /* v8 ignore next */
          ); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.QueryLexer.prototype.emit = function (e) { /* v8 ignore next */ /* v8 ignore next */
          (this.lexemes.push({ /* v8 ignore next */ /* v8 ignore next */
            type: e, /* v8 ignore next */ /* v8 ignore next */
            str: this.sliceString(), /* v8 ignore next */ /* v8 ignore next */
            start: this.start, /* v8 ignore next */ /* v8 ignore next */
            end: this.pos /* v8 ignore next */ /* v8 ignore next */
          }), /* v8 ignore next */ /* v8 ignore next */
            (this.start = this.pos)); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.QueryLexer.prototype.escapeCharacter = function () { /* v8 ignore next */ /* v8 ignore next */
          (this.escapeCharPositions.push(this.pos - 1), (this.pos += 1)); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.QueryLexer.prototype.next = function () { /* v8 ignore next */ /* v8 ignore next */
          if (this.pos >= this.length) return t.QueryLexer.EOS; /* v8 ignore next */ /* v8 ignore next */
          var e = this.str.charAt(this.pos); /* v8 ignore next */ /* v8 ignore next */
          return ((this.pos += 1), e); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.QueryLexer.prototype.width = function () { /* v8 ignore next */ /* v8 ignore next */
          return this.pos - this.start; /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.QueryLexer.prototype.ignore = function () { /* v8 ignore next */ /* v8 ignore next */
          (this.start == this.pos && (this.pos += 1), (this.start = this.pos)); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.QueryLexer.prototype.backup = function () { /* v8 ignore next */ /* v8 ignore next */
          this.pos -= 1; /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.QueryLexer.prototype.acceptDigitRun = function () { /* v8 ignore next */ /* v8 ignore next */
          var e, n; /* v8 ignore next */ /* v8 ignore next */
          do ((e = this.next()), (n = e.charCodeAt(0))); /* v8 ignore next */ /* v8 ignore next */
          while (n > 47 && n < 58); /* v8 ignore next */ /* v8 ignore next */
          e != t.QueryLexer.EOS && this.backup(); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.QueryLexer.prototype.more = function () { /* v8 ignore next */ /* v8 ignore next */
          return this.pos < this.length; /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.QueryLexer.EOS = 'EOS'), /* v8 ignore next */ /* v8 ignore next */
        (t.QueryLexer.FIELD = 'FIELD'), /* v8 ignore next */ /* v8 ignore next */
        (t.QueryLexer.TERM = 'TERM'), /* v8 ignore next */ /* v8 ignore next */
        (t.QueryLexer.EDIT_DISTANCE = 'EDIT_DISTANCE'), /* v8 ignore next */ /* v8 ignore next */
        (t.QueryLexer.BOOST = 'BOOST'), /* v8 ignore next */ /* v8 ignore next */
        (t.QueryLexer.PRESENCE = 'PRESENCE'), /* v8 ignore next */ /* v8 ignore next */
        (t.QueryLexer.lexField = function (e) { /* v8 ignore next */ /* v8 ignore next */
          return (e.backup(), e.emit(t.QueryLexer.FIELD), e.ignore(), t.QueryLexer.lexText); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.QueryLexer.lexTerm = function (e) { /* v8 ignore next */ /* v8 ignore next */
          if ((e.width() > 1 && (e.backup(), e.emit(t.QueryLexer.TERM)), e.ignore(), e.more())) /* v8 ignore next */ /* v8 ignore next */
            return t.QueryLexer.lexText; /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.QueryLexer.lexEditDistance = function (e) { /* v8 ignore next */ /* v8 ignore next */
          return ( /* v8 ignore next */ /* v8 ignore next */
            e.ignore(), /* v8 ignore next */ /* v8 ignore next */
            e.acceptDigitRun(), /* v8 ignore next */ /* v8 ignore next */
            e.emit(t.QueryLexer.EDIT_DISTANCE), /* v8 ignore next */ /* v8 ignore next */
            t.QueryLexer.lexText /* v8 ignore next */ /* v8 ignore next */
          ); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.QueryLexer.lexBoost = function (e) { /* v8 ignore next */ /* v8 ignore next */
          return (e.ignore(), e.acceptDigitRun(), e.emit(t.QueryLexer.BOOST), t.QueryLexer.lexText); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.QueryLexer.lexEOS = function (e) { /* v8 ignore next */ /* v8 ignore next */
          e.width() > 0 && e.emit(t.QueryLexer.TERM); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.QueryLexer.termSeparator = t.tokenizer.separator), /* v8 ignore next */ /* v8 ignore next */
        (t.QueryLexer.lexText = function (e) { /* v8 ignore next */ /* v8 ignore next */
          for (;;) { /* v8 ignore next */ /* v8 ignore next */
            var n = e.next(); /* v8 ignore next */ /* v8 ignore next */
            if (n == t.QueryLexer.EOS) return t.QueryLexer.lexEOS; /* v8 ignore next */ /* v8 ignore next */
            if (n.charCodeAt(0) == 92) { /* v8 ignore next */ /* v8 ignore next */
              e.escapeCharacter(); /* v8 ignore next */ /* v8 ignore next */
              continue; /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
            if (n == ':') return t.QueryLexer.lexField; /* v8 ignore next */ /* v8 ignore next */
            if (n == '~') /* v8 ignore next */ /* v8 ignore next */
              return ( /* v8 ignore next */ /* v8 ignore next */
                e.backup(), /* v8 ignore next */ /* v8 ignore next */
                e.width() > 0 && e.emit(t.QueryLexer.TERM), /* v8 ignore next */ /* v8 ignore next */
                t.QueryLexer.lexEditDistance /* v8 ignore next */ /* v8 ignore next */
              ); /* v8 ignore next */ /* v8 ignore next */
            if (n == '^') /* v8 ignore next */ /* v8 ignore next */
              return ( /* v8 ignore next */ /* v8 ignore next */
                e.backup(), /* v8 ignore next */ /* v8 ignore next */
                e.width() > 0 && e.emit(t.QueryLexer.TERM), /* v8 ignore next */ /* v8 ignore next */
                t.QueryLexer.lexBoost /* v8 ignore next */ /* v8 ignore next */
              ); /* v8 ignore next */ /* v8 ignore next */
            if ((n == '+' && e.width() === 1) || (n == '-' && e.width() === 1)) /* v8 ignore next */ /* v8 ignore next */
              return (e.emit(t.QueryLexer.PRESENCE), t.QueryLexer.lexText); /* v8 ignore next */ /* v8 ignore next */
            if (n.match(t.QueryLexer.termSeparator)) return t.QueryLexer.lexTerm; /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.QueryParser = function (e, n) { /* v8 ignore next */ /* v8 ignore next */
          ((this.lexer = new t.QueryLexer(e)), /* v8 ignore next */ /* v8 ignore next */
            (this.query = n), /* v8 ignore next */ /* v8 ignore next */
            (this.currentClause = {}), /* v8 ignore next */ /* v8 ignore next */
            (this.lexemeIdx = 0)); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.QueryParser.prototype.parse = function () { /* v8 ignore next */ /* v8 ignore next */
          (this.lexer.run(), (this.lexemes = this.lexer.lexemes)); /* v8 ignore next */ /* v8 ignore next */
          for (var e = t.QueryParser.parseClause; e; ) e = e(this); /* v8 ignore next */ /* v8 ignore next */
          return this.query; /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.QueryParser.prototype.peekLexeme = function () { /* v8 ignore next */ /* v8 ignore next */
          return this.lexemes[this.lexemeIdx]; /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.QueryParser.prototype.consumeLexeme = function () { /* v8 ignore next */ /* v8 ignore next */
          var e = this.peekLexeme(); /* v8 ignore next */ /* v8 ignore next */
          return ((this.lexemeIdx += 1), e); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.QueryParser.prototype.nextClause = function () { /* v8 ignore next */ /* v8 ignore next */
          var e = this.currentClause; /* v8 ignore next */ /* v8 ignore next */
          (this.query.clause(e), (this.currentClause = {})); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.QueryParser.parseClause = function (e) { /* v8 ignore next */ /* v8 ignore next */
          var n = e.peekLexeme(); /* v8 ignore next */ /* v8 ignore next */
          if (n != null) /* v8 ignore next */ /* v8 ignore next */
            switch (n.type) { /* v8 ignore next */ /* v8 ignore next */
              case t.QueryLexer.PRESENCE: /* v8 ignore next */ /* v8 ignore next */
                return t.QueryParser.parsePresence; /* v8 ignore next */ /* v8 ignore next */
              case t.QueryLexer.FIELD: /* v8 ignore next */ /* v8 ignore next */
                return t.QueryParser.parseField; /* v8 ignore next */ /* v8 ignore next */
              case t.QueryLexer.TERM: /* v8 ignore next */ /* v8 ignore next */
                return t.QueryParser.parseTerm; /* v8 ignore next */ /* v8 ignore next */
              default: /* v8 ignore next */ /* v8 ignore next */
                var r = 'expected either a field or a term, found ' + n.type; /* v8 ignore next */ /* v8 ignore next */
                throw ( /* v8 ignore next */ /* v8 ignore next */
                  n.str.length >= 1 && (r += " with value '" + n.str + "'"), /* v8 ignore next */ /* v8 ignore next */
                  new t.QueryParseError(r, n.start, n.end) /* v8 ignore next */ /* v8 ignore next */
                ); /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.QueryParser.parsePresence = function (e) { /* v8 ignore next */ /* v8 ignore next */
          var n = e.consumeLexeme(); /* v8 ignore next */ /* v8 ignore next */
          if (n != null) { /* v8 ignore next */ /* v8 ignore next */
            switch (n.str) { /* v8 ignore next */ /* v8 ignore next */
              case '-': /* v8 ignore next */ /* v8 ignore next */
                e.currentClause.presence = t.Query.presence.PROHIBITED; /* v8 ignore next */ /* v8 ignore next */
                break; /* v8 ignore next */ /* v8 ignore next */
              case '+': /* v8 ignore next */ /* v8 ignore next */
                e.currentClause.presence = t.Query.presence.REQUIRED; /* v8 ignore next */ /* v8 ignore next */
                break; /* v8 ignore next */ /* v8 ignore next */
              default: /* v8 ignore next */ /* v8 ignore next */
                var r = "unrecognised presence operator'" + n.str + "'"; /* v8 ignore next */ /* v8 ignore next */
                throw new t.QueryParseError(r, n.start, n.end); /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
            var i = e.peekLexeme(); /* v8 ignore next */ /* v8 ignore next */
            if (i == null) { /* v8 ignore next */ /* v8 ignore next */
              var r = 'expecting term or field, found nothing'; /* v8 ignore next */ /* v8 ignore next */
              throw new t.QueryParseError(r, n.start, n.end); /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
            switch (i.type) { /* v8 ignore next */ /* v8 ignore next */
              case t.QueryLexer.FIELD: /* v8 ignore next */ /* v8 ignore next */
                return t.QueryParser.parseField; /* v8 ignore next */ /* v8 ignore next */
              case t.QueryLexer.TERM: /* v8 ignore next */ /* v8 ignore next */
                return t.QueryParser.parseTerm; /* v8 ignore next */ /* v8 ignore next */
              default: /* v8 ignore next */ /* v8 ignore next */
                var r = "expecting term or field, found '" + i.type + "'"; /* v8 ignore next */ /* v8 ignore next */
                throw new t.QueryParseError(r, i.start, i.end); /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.QueryParser.parseField = function (e) { /* v8 ignore next */ /* v8 ignore next */
          var n = e.consumeLexeme(); /* v8 ignore next */ /* v8 ignore next */
          if (n != null) { /* v8 ignore next */ /* v8 ignore next */
            if (e.query.allFields.indexOf(n.str) == -1) { /* v8 ignore next */ /* v8 ignore next */
              var r = e.query.allFields /* v8 ignore next */ /* v8 ignore next */
                  .map(function (o) { /* v8 ignore next */ /* v8 ignore next */
                    return "'" + o + "'"; /* v8 ignore next */ /* v8 ignore next */
                  }) /* v8 ignore next */ /* v8 ignore next */
                  .join(', '), /* v8 ignore next */ /* v8 ignore next */
                i = "unrecognised field '" + n.str + "', possible fields: " + r; /* v8 ignore next */ /* v8 ignore next */
              throw new t.QueryParseError(i, n.start, n.end); /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
            e.currentClause.fields = [n.str]; /* v8 ignore next */ /* v8 ignore next */
            var s = e.peekLexeme(); /* v8 ignore next */ /* v8 ignore next */
            if (s == null) { /* v8 ignore next */ /* v8 ignore next */
              var i = 'expecting term, found nothing'; /* v8 ignore next */ /* v8 ignore next */
              throw new t.QueryParseError(i, n.start, n.end); /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
            switch (s.type) { /* v8 ignore next */ /* v8 ignore next */
              case t.QueryLexer.TERM: /* v8 ignore next */ /* v8 ignore next */
                return t.QueryParser.parseTerm; /* v8 ignore next */ /* v8 ignore next */
              default: /* v8 ignore next */ /* v8 ignore next */
                var i = "expecting term, found '" + s.type + "'"; /* v8 ignore next */ /* v8 ignore next */
                throw new t.QueryParseError(i, s.start, s.end); /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.QueryParser.parseTerm = function (e) { /* v8 ignore next */ /* v8 ignore next */
          var n = e.consumeLexeme(); /* v8 ignore next */ /* v8 ignore next */
          if (n != null) { /* v8 ignore next */ /* v8 ignore next */
            ((e.currentClause.term = n.str.toLowerCase()), /* v8 ignore next */ /* v8 ignore next */
              n.str.indexOf('*') != -1 && (e.currentClause.usePipeline = !1)); /* v8 ignore next */ /* v8 ignore next */
            var r = e.peekLexeme(); /* v8 ignore next */ /* v8 ignore next */
            if (r == null) { /* v8 ignore next */ /* v8 ignore next */
              e.nextClause(); /* v8 ignore next */ /* v8 ignore next */
              return; /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
            switch (r.type) { /* v8 ignore next */ /* v8 ignore next */
              case t.QueryLexer.TERM: /* v8 ignore next */ /* v8 ignore next */
                return (e.nextClause(), t.QueryParser.parseTerm); /* v8 ignore next */ /* v8 ignore next */
              case t.QueryLexer.FIELD: /* v8 ignore next */ /* v8 ignore next */
                return (e.nextClause(), t.QueryParser.parseField); /* v8 ignore next */ /* v8 ignore next */
              case t.QueryLexer.EDIT_DISTANCE: /* v8 ignore next */ /* v8 ignore next */
                return t.QueryParser.parseEditDistance; /* v8 ignore next */ /* v8 ignore next */
              case t.QueryLexer.BOOST: /* v8 ignore next */ /* v8 ignore next */
                return t.QueryParser.parseBoost; /* v8 ignore next */ /* v8 ignore next */
              case t.QueryLexer.PRESENCE: /* v8 ignore next */ /* v8 ignore next */
                return (e.nextClause(), t.QueryParser.parsePresence); /* v8 ignore next */ /* v8 ignore next */
              default: /* v8 ignore next */ /* v8 ignore next */
                var i = "Unexpected lexeme type '" + r.type + "'"; /* v8 ignore next */ /* v8 ignore next */
                throw new t.QueryParseError(i, r.start, r.end); /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.QueryParser.parseEditDistance = function (e) { /* v8 ignore next */ /* v8 ignore next */
          var n = e.consumeLexeme(); /* v8 ignore next */ /* v8 ignore next */
          if (n != null) { /* v8 ignore next */ /* v8 ignore next */
            var r = parseInt(n.str, 10); /* v8 ignore next */ /* v8 ignore next */
            if (isNaN(r)) { /* v8 ignore next */ /* v8 ignore next */
              var i = 'edit distance must be numeric'; /* v8 ignore next */ /* v8 ignore next */
              throw new t.QueryParseError(i, n.start, n.end); /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
            e.currentClause.editDistance = r; /* v8 ignore next */ /* v8 ignore next */
            var s = e.peekLexeme(); /* v8 ignore next */ /* v8 ignore next */
            if (s == null) { /* v8 ignore next */ /* v8 ignore next */
              e.nextClause(); /* v8 ignore next */ /* v8 ignore next */
              return; /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
            switch (s.type) { /* v8 ignore next */ /* v8 ignore next */
              case t.QueryLexer.TERM: /* v8 ignore next */ /* v8 ignore next */
                return (e.nextClause(), t.QueryParser.parseTerm); /* v8 ignore next */ /* v8 ignore next */
              case t.QueryLexer.FIELD: /* v8 ignore next */ /* v8 ignore next */
                return (e.nextClause(), t.QueryParser.parseField); /* v8 ignore next */ /* v8 ignore next */
              case t.QueryLexer.EDIT_DISTANCE: /* v8 ignore next */ /* v8 ignore next */
                return t.QueryParser.parseEditDistance; /* v8 ignore next */ /* v8 ignore next */
              case t.QueryLexer.BOOST: /* v8 ignore next */ /* v8 ignore next */
                return t.QueryParser.parseBoost; /* v8 ignore next */ /* v8 ignore next */
              case t.QueryLexer.PRESENCE: /* v8 ignore next */ /* v8 ignore next */
                return (e.nextClause(), t.QueryParser.parsePresence); /* v8 ignore next */ /* v8 ignore next */
              default: /* v8 ignore next */ /* v8 ignore next */
                var i = "Unexpected lexeme type '" + s.type + "'"; /* v8 ignore next */ /* v8 ignore next */
                throw new t.QueryParseError(i, s.start, s.end); /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (t.QueryParser.parseBoost = function (e) { /* v8 ignore next */ /* v8 ignore next */
          var n = e.consumeLexeme(); /* v8 ignore next */ /* v8 ignore next */
          if (n != null) { /* v8 ignore next */ /* v8 ignore next */
            var r = parseInt(n.str, 10); /* v8 ignore next */ /* v8 ignore next */
            if (isNaN(r)) { /* v8 ignore next */ /* v8 ignore next */
              var i = 'boost must be numeric'; /* v8 ignore next */ /* v8 ignore next */
              throw new t.QueryParseError(i, n.start, n.end); /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
            e.currentClause.boost = r; /* v8 ignore next */ /* v8 ignore next */
            var s = e.peekLexeme(); /* v8 ignore next */ /* v8 ignore next */
            if (s == null) { /* v8 ignore next */ /* v8 ignore next */
              e.nextClause(); /* v8 ignore next */ /* v8 ignore next */
              return; /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
            switch (s.type) { /* v8 ignore next */ /* v8 ignore next */
              case t.QueryLexer.TERM: /* v8 ignore next */ /* v8 ignore next */
                return (e.nextClause(), t.QueryParser.parseTerm); /* v8 ignore next */ /* v8 ignore next */
              case t.QueryLexer.FIELD: /* v8 ignore next */ /* v8 ignore next */
                return (e.nextClause(), t.QueryParser.parseField); /* v8 ignore next */ /* v8 ignore next */
              case t.QueryLexer.EDIT_DISTANCE: /* v8 ignore next */ /* v8 ignore next */
                return t.QueryParser.parseEditDistance; /* v8 ignore next */ /* v8 ignore next */
              case t.QueryLexer.BOOST: /* v8 ignore next */ /* v8 ignore next */
                return t.QueryParser.parseBoost; /* v8 ignore next */ /* v8 ignore next */
              case t.QueryLexer.PRESENCE: /* v8 ignore next */ /* v8 ignore next */
                return (e.nextClause(), t.QueryParser.parsePresence); /* v8 ignore next */ /* v8 ignore next */
              default: /* v8 ignore next */ /* v8 ignore next */
                var i = "Unexpected lexeme type '" + s.type + "'"; /* v8 ignore next */ /* v8 ignore next */
                throw new t.QueryParseError(i, s.start, s.end); /* v8 ignore next */ /* v8 ignore next */
            } /* v8 ignore next */ /* v8 ignore next */
          } /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        (function (e, n) { /* v8 ignore next */ /* v8 ignore next */
          typeof define == 'function' && define.amd /* v8 ignore next */ /* v8 ignore next */
            ? define(n) /* v8 ignore next */ /* v8 ignore next */
            : typeof se == 'object' /* v8 ignore next */ /* v8 ignore next */
              ? (oe.exports = n()) /* v8 ignore next */ /* v8 ignore next */
              : (e.lunr = n()); /* v8 ignore next */ /* v8 ignore next */
        })(this, function () { /* v8 ignore next */ /* v8 ignore next */
          return t; /* v8 ignore next */ /* v8 ignore next */
        })); /* v8 ignore next */ /* v8 ignore next */
    })(); /* v8 ignore next */ /* v8 ignore next */
  }); /* v8 ignore next */ /* v8 ignore next */
  var re = []; /* v8 ignore next */ /* v8 ignore next */
  function G(t, e) { /* v8 ignore next */ /* v8 ignore next */
    re.push({ selector: e, constructor: t }); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  var U = class { /* v8 ignore next */ /* v8 ignore next */
    constructor() { /* v8 ignore next */ /* v8 ignore next */
      this.alwaysVisibleMember = null; /* v8 ignore next */ /* v8 ignore next */
      (this.createComponents(document.body), /* v8 ignore next */ /* v8 ignore next */
        this.ensureFocusedElementVisible(), /* v8 ignore next */ /* v8 ignore next */
        this.listenForCodeCopies(), /* v8 ignore next */ /* v8 ignore next */
        window.addEventListener('hashchange', () => this.ensureFocusedElementVisible()), /* v8 ignore next */ /* v8 ignore next */
        document.body.style.display || /* v8 ignore next */ /* v8 ignore next */
          (this.ensureFocusedElementVisible(), this.updateIndexVisibility(), this.scrollToHash())); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    createComponents(e) { /* v8 ignore next */ /* v8 ignore next */
      re.forEach((n) => { /* v8 ignore next */ /* v8 ignore next */
        e.querySelectorAll(n.selector).forEach((r) => { /* v8 ignore next */ /* v8 ignore next */
          r.dataset.hasInstance || /* v8 ignore next */ /* v8 ignore next */
            (new n.constructor({ el: r, app: this }), (r.dataset.hasInstance = String(!0))); /* v8 ignore next */ /* v8 ignore next */
        }); /* v8 ignore next */ /* v8 ignore next */
      }); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    filterChanged() { /* v8 ignore next */ /* v8 ignore next */
      this.ensureFocusedElementVisible(); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    showPage() { /* v8 ignore next */ /* v8 ignore next */
      document.body.style.display && /* v8 ignore next */ /* v8 ignore next */
        (console.log('Show page'), /* v8 ignore next */ /* v8 ignore next */
        document.body.style.removeProperty('display'), /* v8 ignore next */ /* v8 ignore next */
        this.ensureFocusedElementVisible(), /* v8 ignore next */ /* v8 ignore next */
        this.updateIndexVisibility(), /* v8 ignore next */ /* v8 ignore next */
        this.scrollToHash()); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    scrollToHash() { /* v8 ignore next */ /* v8 ignore next */
      if (location.hash) { /* v8 ignore next */ /* v8 ignore next */
        console.log('Scorlling'); /* v8 ignore next */ /* v8 ignore next */
        let e = document.getElementById(location.hash.substring(1)); /* v8 ignore next */ /* v8 ignore next */
        if (!e) return; /* v8 ignore next */ /* v8 ignore next */
        e.scrollIntoView({ behavior: 'instant', block: 'start' }); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    ensureActivePageVisible() { /* v8 ignore next */ /* v8 ignore next */
      let e = document.querySelector('.tsd-navigation .current'), /* v8 ignore next */ /* v8 ignore next */
        n = e?.parentElement; /* v8 ignore next */ /* v8 ignore next */
      for (; n && !n.classList.contains('.tsd-navigation'); ) /* v8 ignore next */ /* v8 ignore next */
        (n instanceof HTMLDetailsElement && (n.open = !0), (n = n.parentElement)); /* v8 ignore next */ /* v8 ignore next */
      if (e && !e.checkVisibility()) { /* v8 ignore next */ /* v8 ignore next */
        let r = e.getBoundingClientRect().top - document.documentElement.clientHeight / 4; /* v8 ignore next */ /* v8 ignore next */
        document.querySelector('.site-menu').scrollTop = r; /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    updateIndexVisibility() { /* v8 ignore next */ /* v8 ignore next */
      let e = document.querySelector('.tsd-index-content'), /* v8 ignore next */ /* v8 ignore next */
        n = e?.open; /* v8 ignore next */ /* v8 ignore next */
      (e && (e.open = !0), /* v8 ignore next */ /* v8 ignore next */
        document.querySelectorAll('.tsd-index-section').forEach((r) => { /* v8 ignore next */ /* v8 ignore next */
          r.style.display = 'block'; /* v8 ignore next */ /* v8 ignore next */
          let i = Array.from(r.querySelectorAll('.tsd-index-link')).every( /* v8 ignore next */ /* v8 ignore next */
            (s) => s.offsetParent == null /* v8 ignore next */ /* v8 ignore next */
          ); /* v8 ignore next */ /* v8 ignore next */
          r.style.display = i ? 'none' : 'block'; /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        e && (e.open = n)); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    ensureFocusedElementVisible() { /* v8 ignore next */ /* v8 ignore next */
      if ( /* v8 ignore next */ /* v8 ignore next */
        (this.alwaysVisibleMember && /* v8 ignore next */ /* v8 ignore next */
          (this.alwaysVisibleMember.classList.remove('always-visible'), /* v8 ignore next */ /* v8 ignore next */
          this.alwaysVisibleMember.firstElementChild.remove(), /* v8 ignore next */ /* v8 ignore next */
          (this.alwaysVisibleMember = null)), /* v8 ignore next */ /* v8 ignore next */
        !location.hash) /* v8 ignore next */ /* v8 ignore next */
      ) /* v8 ignore next */ /* v8 ignore next */
        return; /* v8 ignore next */ /* v8 ignore next */
      let e = document.getElementById(location.hash.substring(1)); /* v8 ignore next */ /* v8 ignore next */
      if (!e) return; /* v8 ignore next */ /* v8 ignore next */
      let n = e.parentElement; /* v8 ignore next */ /* v8 ignore next */
      for (; n && n.tagName !== 'SECTION'; ) n = n.parentElement; /* v8 ignore next */ /* v8 ignore next */
      if (n && n.offsetParent == null) { /* v8 ignore next */ /* v8 ignore next */
        ((this.alwaysVisibleMember = n), n.classList.add('always-visible')); /* v8 ignore next */ /* v8 ignore next */
        let r = document.createElement('p'); /* v8 ignore next */ /* v8 ignore next */
        (r.classList.add('warning'), /* v8 ignore next */ /* v8 ignore next */
          (r.textContent = 'This member is normally hidden due to your filter settings.'), /* v8 ignore next */ /* v8 ignore next */
          n.prepend(r)); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    listenForCodeCopies() { /* v8 ignore next */ /* v8 ignore next */
      document.querySelectorAll('pre > button').forEach((e) => { /* v8 ignore next */ /* v8 ignore next */
        let n; /* v8 ignore next */ /* v8 ignore next */
        e.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
          (e.previousElementSibling instanceof HTMLElement && /* v8 ignore next */ /* v8 ignore next */
            navigator.clipboard.writeText(e.previousElementSibling.innerText.trim()), /* v8 ignore next */ /* v8 ignore next */
            (e.textContent = 'Copied!'), /* v8 ignore next */ /* v8 ignore next */
            e.classList.add('visible'), /* v8 ignore next */ /* v8 ignore next */
            clearTimeout(n), /* v8 ignore next */ /* v8 ignore next */
            (n = setTimeout(() => { /* v8 ignore next */ /* v8 ignore next */
              (e.classList.remove('visible'), /* v8 ignore next */ /* v8 ignore next */
                (n = setTimeout(() => { /* v8 ignore next */ /* v8 ignore next */
                  e.textContent = 'Copy'; /* v8 ignore next */ /* v8 ignore next */
                }, 100))); /* v8 ignore next */ /* v8 ignore next */
            }, 1e3))); /* v8 ignore next */ /* v8 ignore next */
        }); /* v8 ignore next */ /* v8 ignore next */
      }); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  }; /* v8 ignore next */ /* v8 ignore next */
  var ie = (t, e = 100) => { /* v8 ignore next */ /* v8 ignore next */
    let n; /* v8 ignore next */ /* v8 ignore next */
    return () => { /* v8 ignore next */ /* v8 ignore next */
      (clearTimeout(n), (n = setTimeout(() => t(), e))); /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
  }; /* v8 ignore next */ /* v8 ignore next */
  var de = De(ae()); /* v8 ignore next */ /* v8 ignore next */
  async function le(t, e) { /* v8 ignore next */ /* v8 ignore next */
    if (!window.searchData) return; /* v8 ignore next */ /* v8 ignore next */
    let n = await fetch(window.searchData), /* v8 ignore next */ /* v8 ignore next */
      r = new Blob([await n.arrayBuffer()]).stream().pipeThrough(new DecompressionStream('gzip')), /* v8 ignore next */ /* v8 ignore next */
      i = await new Response(r).json(); /* v8 ignore next */ /* v8 ignore next */
    ((t.data = i), /* v8 ignore next */ /* v8 ignore next */
      (t.index = de.Index.load(i.index)), /* v8 ignore next */ /* v8 ignore next */
      e.classList.remove('loading'), /* v8 ignore next */ /* v8 ignore next */
      e.classList.add('ready')); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  function he() { /* v8 ignore next */ /* v8 ignore next */
    let t = document.getElementById('tsd-search'); /* v8 ignore next */ /* v8 ignore next */
    if (!t) return; /* v8 ignore next */ /* v8 ignore next */
    let e = { base: t.dataset.base + '/' }, /* v8 ignore next */ /* v8 ignore next */
      n = document.getElementById('tsd-search-script'); /* v8 ignore next */ /* v8 ignore next */
    (t.classList.add('loading'), /* v8 ignore next */ /* v8 ignore next */
      n && /* v8 ignore next */ /* v8 ignore next */
        (n.addEventListener('error', () => { /* v8 ignore next */ /* v8 ignore next */
          (t.classList.remove('loading'), t.classList.add('failure')); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        n.addEventListener('load', () => { /* v8 ignore next */ /* v8 ignore next */
          le(e, t); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        le(e, t))); /* v8 ignore next */ /* v8 ignore next */
    let r = document.querySelector('#tsd-search input'), /* v8 ignore next */ /* v8 ignore next */
      i = document.querySelector('#tsd-search .results'); /* v8 ignore next */ /* v8 ignore next */
    if (!r || !i) throw new Error('The input field or the result list wrapper was not found'); /* v8 ignore next */ /* v8 ignore next */
    let s = !1; /* v8 ignore next */ /* v8 ignore next */
    (i.addEventListener('mousedown', () => (s = !0)), /* v8 ignore next */ /* v8 ignore next */
      i.addEventListener('mouseup', () => { /* v8 ignore next */ /* v8 ignore next */
        ((s = !1), t.classList.remove('has-focus')); /* v8 ignore next */ /* v8 ignore next */
      }), /* v8 ignore next */ /* v8 ignore next */
      r.addEventListener('focus', () => t.classList.add('has-focus')), /* v8 ignore next */ /* v8 ignore next */
      r.addEventListener('blur', () => { /* v8 ignore next */ /* v8 ignore next */
        s || ((s = !1), t.classList.remove('has-focus')); /* v8 ignore next */ /* v8 ignore next */
      }), /* v8 ignore next */ /* v8 ignore next */
      Ae(t, i, r, e)); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  function Ae(t, e, n, r) { /* v8 ignore next */ /* v8 ignore next */
    n.addEventListener( /* v8 ignore next */ /* v8 ignore next */
      'input', /* v8 ignore next */ /* v8 ignore next */
      ie(() => { /* v8 ignore next */ /* v8 ignore next */
        Ve(t, e, n, r); /* v8 ignore next */ /* v8 ignore next */
      }, 200) /* v8 ignore next */ /* v8 ignore next */
    ); /* v8 ignore next */ /* v8 ignore next */
    let i = !1; /* v8 ignore next */ /* v8 ignore next */
    (n.addEventListener('keydown', (s) => { /* v8 ignore next */ /* v8 ignore next */
      ((i = !0), /* v8 ignore next */ /* v8 ignore next */
        s.key == 'Enter' /* v8 ignore next */ /* v8 ignore next */
          ? Ne(e, n) /* v8 ignore next */ /* v8 ignore next */
          : s.key == 'Escape' /* v8 ignore next */ /* v8 ignore next */
            ? n.blur() /* v8 ignore next */ /* v8 ignore next */
            : s.key == 'ArrowUp' /* v8 ignore next */ /* v8 ignore next */
              ? ue(e, -1) /* v8 ignore next */ /* v8 ignore next */
              : s.key === 'ArrowDown' /* v8 ignore next */ /* v8 ignore next */
                ? ue(e, 1) /* v8 ignore next */ /* v8 ignore next */
                : (i = !1)); /* v8 ignore next */ /* v8 ignore next */
    }), /* v8 ignore next */ /* v8 ignore next */
      n.addEventListener('keypress', (s) => { /* v8 ignore next */ /* v8 ignore next */
        i && s.preventDefault(); /* v8 ignore next */ /* v8 ignore next */
      }), /* v8 ignore next */ /* v8 ignore next */
      document.body.addEventListener('keydown', (s) => { /* v8 ignore next */ /* v8 ignore next */
        s.altKey || /* v8 ignore next */ /* v8 ignore next */
          s.ctrlKey || /* v8 ignore next */ /* v8 ignore next */
          s.metaKey || /* v8 ignore next */ /* v8 ignore next */
          (!n.matches(':focus') && s.key === '/' && (n.focus(), s.preventDefault())); /* v8 ignore next */ /* v8 ignore next */
      })); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  function Ve(t, e, n, r) { /* v8 ignore next */ /* v8 ignore next */
    if (!r.index || !r.data) return; /* v8 ignore next */ /* v8 ignore next */
    e.textContent = ''; /* v8 ignore next */ /* v8 ignore next */
    let i = n.value.trim(), /* v8 ignore next */ /* v8 ignore next */
      s; /* v8 ignore next */ /* v8 ignore next */
    if (i) { /* v8 ignore next */ /* v8 ignore next */
      let o = i /* v8 ignore next */ /* v8 ignore next */
        .split(' ') /* v8 ignore next */ /* v8 ignore next */
        .map((a) => (a.length ? `*${a}*` : '')) /* v8 ignore next */ /* v8 ignore next */
        .join(' '); /* v8 ignore next */ /* v8 ignore next */
      s = r.index.search(o); /* v8 ignore next */ /* v8 ignore next */
    } else s = []; /* v8 ignore next */ /* v8 ignore next */
    for (let o = 0; o < s.length; o++) { /* v8 ignore next */ /* v8 ignore next */
      let a = s[o], /* v8 ignore next */ /* v8 ignore next */
        l = r.data.rows[Number(a.ref)], /* v8 ignore next */ /* v8 ignore next */
        u = 1; /* v8 ignore next */ /* v8 ignore next */
      (l.name.toLowerCase().startsWith(i.toLowerCase()) && /* v8 ignore next */ /* v8 ignore next */
        (u *= 1 + 1 / (1 + Math.abs(l.name.length - i.length))), /* v8 ignore next */ /* v8 ignore next */
        (a.score *= u)); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    if (s.length === 0) { /* v8 ignore next */ /* v8 ignore next */
      let o = document.createElement('li'); /* v8 ignore next */ /* v8 ignore next */
      o.classList.add('no-results'); /* v8 ignore next */ /* v8 ignore next */
      let a = document.createElement('span'); /* v8 ignore next */ /* v8 ignore next */
      ((a.textContent = 'No results found'), o.appendChild(a), e.appendChild(o)); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    s.sort((o, a) => a.score - o.score); /* v8 ignore next */ /* v8 ignore next */
    for (let o = 0, a = Math.min(10, s.length); o < a; o++) { /* v8 ignore next */ /* v8 ignore next */
      let l = r.data.rows[Number(s[o].ref)], /* v8 ignore next */ /* v8 ignore next */
        u = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" class="tsd-kind-icon"><use href="#icon-${l.kind}"></use></svg>`, /* v8 ignore next */ /* v8 ignore next */
        d = ce(l.name, i); /* v8 ignore next */ /* v8 ignore next */
      (globalThis.DEBUG_SEARCH_WEIGHTS && (d += ` (score: ${s[o].score.toFixed(2)})`), /* v8 ignore next */ /* v8 ignore next */
        l.parent && /* v8 ignore next */ /* v8 ignore next */
          (d = `<span class="parent"> /* v8 ignore next */ /* v8 ignore next */
                ${ce(l.parent, i)}.</span>${d}`)); /* v8 ignore next */ /* v8 ignore next */
      let y = document.createElement('li'); /* v8 ignore next */ /* v8 ignore next */
      y.classList.value = l.classes ?? ''; /* v8 ignore next */ /* v8 ignore next */
      let p = document.createElement('a'); /* v8 ignore next */ /* v8 ignore next */
      ((p.href = r.base + l.url), (p.innerHTML = u + d), y.append(p), e.appendChild(y)); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  function ue(t, e) { /* v8 ignore next */ /* v8 ignore next */
    let n = t.querySelector('.current'); /* v8 ignore next */ /* v8 ignore next */
    if (!n) /* v8 ignore next */ /* v8 ignore next */
      ((n = t.querySelector(e == 1 ? 'li:first-child' : 'li:last-child')), /* v8 ignore next */ /* v8 ignore next */
        n && n.classList.add('current')); /* v8 ignore next */ /* v8 ignore next */
    else { /* v8 ignore next */ /* v8 ignore next */
      let r = n; /* v8 ignore next */ /* v8 ignore next */
      if (e === 1) /* v8 ignore next */ /* v8 ignore next */
        do r = r.nextElementSibling ?? void 0; /* v8 ignore next */ /* v8 ignore next */
        while (r instanceof HTMLElement && r.offsetParent == null); /* v8 ignore next */ /* v8 ignore next */
      else /* v8 ignore next */ /* v8 ignore next */
        do r = r.previousElementSibling ?? void 0; /* v8 ignore next */ /* v8 ignore next */
        while (r instanceof HTMLElement && r.offsetParent == null); /* v8 ignore next */ /* v8 ignore next */
      r && (n.classList.remove('current'), r.classList.add('current')); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  function Ne(t, e) { /* v8 ignore next */ /* v8 ignore next */
    let n = t.querySelector('.current'); /* v8 ignore next */ /* v8 ignore next */
    if ((n || (n = t.querySelector('li:first-child')), n)) { /* v8 ignore next */ /* v8 ignore next */
      let r = n.querySelector('a'); /* v8 ignore next */ /* v8 ignore next */
      (r && (window.location.href = r.href), e.blur()); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  function ce(t, e) { /* v8 ignore next */ /* v8 ignore next */
    if (e === '') return t; /* v8 ignore next */ /* v8 ignore next */
    let n = t.toLocaleLowerCase(), /* v8 ignore next */ /* v8 ignore next */
      r = e.toLocaleLowerCase(), /* v8 ignore next */ /* v8 ignore next */
      i = [], /* v8 ignore next */ /* v8 ignore next */
      s = 0, /* v8 ignore next */ /* v8 ignore next */
      o = n.indexOf(r); /* v8 ignore next */ /* v8 ignore next */
    for (; o != -1; ) /* v8 ignore next */ /* v8 ignore next */
      (i.push(K(t.substring(s, o)), `<b>${K(t.substring(o, o + r.length))}</b>`), /* v8 ignore next */ /* v8 ignore next */
        (s = o + r.length), /* v8 ignore next */ /* v8 ignore next */
        (o = n.indexOf(r, s))); /* v8 ignore next */ /* v8 ignore next */
    return (i.push(K(t.substring(s))), i.join('')); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  var He = { '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#039;', '"': '&quot;' }; /* v8 ignore next */ /* v8 ignore next */
  function K(t) { /* v8 ignore next */ /* v8 ignore next */
    return t.replace(/[&<>"'"]/g, (e) => He[e]); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  var I = class { /* v8 ignore next */ /* v8 ignore next */
    constructor(e) { /* v8 ignore next */ /* v8 ignore next */
      ((this.el = e.el), (this.app = e.app)); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  }; /* v8 ignore next */ /* v8 ignore next */
  var F = 'mousedown', /* v8 ignore next */ /* v8 ignore next */
    fe = 'mousemove', /* v8 ignore next */ /* v8 ignore next */
    H = 'mouseup', /* v8 ignore next */ /* v8 ignore next */
    J = { x: 0, y: 0 }, /* v8 ignore next */ /* v8 ignore next */
    pe = !1, /* v8 ignore next */ /* v8 ignore next */
    ee = !1, /* v8 ignore next */ /* v8 ignore next */
    Be = !1, /* v8 ignore next */ /* v8 ignore next */
    D = !1, /* v8 ignore next */ /* v8 ignore next */
    me = /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent); /* v8 ignore next */ /* v8 ignore next */
  document.documentElement.classList.add(me ? 'is-mobile' : 'not-mobile'); /* v8 ignore next */ /* v8 ignore next */
  me && /* v8 ignore next */ /* v8 ignore next */
    'ontouchstart' in document.documentElement && /* v8 ignore next */ /* v8 ignore next */
    ((Be = !0), (F = 'touchstart'), (fe = 'touchmove'), (H = 'touchend')); /* v8 ignore next */ /* v8 ignore next */
  document.addEventListener(F, (t) => { /* v8 ignore next */ /* v8 ignore next */
    ((ee = !0), (D = !1)); /* v8 ignore next */ /* v8 ignore next */
    let e = F == 'touchstart' ? t.targetTouches[0] : t; /* v8 ignore next */ /* v8 ignore next */
    ((J.y = e.pageY || 0), (J.x = e.pageX || 0)); /* v8 ignore next */ /* v8 ignore next */
  }); /* v8 ignore next */ /* v8 ignore next */
  document.addEventListener(fe, (t) => { /* v8 ignore next */ /* v8 ignore next */
    if (ee && !D) { /* v8 ignore next */ /* v8 ignore next */
      let e = F == 'touchstart' ? t.targetTouches[0] : t, /* v8 ignore next */ /* v8 ignore next */
        n = J.x - (e.pageX || 0), /* v8 ignore next */ /* v8 ignore next */
        r = J.y - (e.pageY || 0); /* v8 ignore next */ /* v8 ignore next */
      D = Math.sqrt(n * n + r * r) > 10; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  }); /* v8 ignore next */ /* v8 ignore next */
  document.addEventListener(H, () => { /* v8 ignore next */ /* v8 ignore next */
    ee = !1; /* v8 ignore next */ /* v8 ignore next */
  }); /* v8 ignore next */ /* v8 ignore next */
  document.addEventListener('click', (t) => { /* v8 ignore next */ /* v8 ignore next */
    pe && (t.preventDefault(), t.stopImmediatePropagation(), (pe = !1)); /* v8 ignore next */ /* v8 ignore next */
  }); /* v8 ignore next */ /* v8 ignore next */
  var X = class extends I { /* v8 ignore next */ /* v8 ignore next */
    constructor(e) { /* v8 ignore next */ /* v8 ignore next */
      (super(e), /* v8 ignore next */ /* v8 ignore next */
        (this.className = this.el.dataset.toggle || ''), /* v8 ignore next */ /* v8 ignore next */
        this.el.addEventListener(H, (n) => this.onPointerUp(n)), /* v8 ignore next */ /* v8 ignore next */
        this.el.addEventListener('click', (n) => n.preventDefault()), /* v8 ignore next */ /* v8 ignore next */
        document.addEventListener(F, (n) => this.onDocumentPointerDown(n)), /* v8 ignore next */ /* v8 ignore next */
        document.addEventListener(H, (n) => this.onDocumentPointerUp(n))); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    setActive(e) { /* v8 ignore next */ /* v8 ignore next */
      if (this.active == e) return; /* v8 ignore next */ /* v8 ignore next */
      ((this.active = e), /* v8 ignore next */ /* v8 ignore next */
        document.documentElement.classList.toggle('has-' + this.className, e), /* v8 ignore next */ /* v8 ignore next */
        this.el.classList.toggle('active', e)); /* v8 ignore next */ /* v8 ignore next */
      let n = (this.active ? 'to-has-' : 'from-has-') + this.className; /* v8 ignore next */ /* v8 ignore next */
      (document.documentElement.classList.add(n), /* v8 ignore next */ /* v8 ignore next */
        setTimeout(() => document.documentElement.classList.remove(n), 500)); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    onPointerUp(e) { /* v8 ignore next */ /* v8 ignore next */
      D || (this.setActive(!0), e.preventDefault()); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    onDocumentPointerDown(e) { /* v8 ignore next */ /* v8 ignore next */
      if (this.active) { /* v8 ignore next */ /* v8 ignore next */
        if (e.target.closest('.col-sidebar, .tsd-filter-group')) return; /* v8 ignore next */ /* v8 ignore next */
        this.setActive(!1); /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    onDocumentPointerUp(e) { /* v8 ignore next */ /* v8 ignore next */
      if (!D && this.active && e.target.closest('.col-sidebar')) { /* v8 ignore next */ /* v8 ignore next */
        let n = e.target.closest('a'); /* v8 ignore next */ /* v8 ignore next */
        if (n) { /* v8 ignore next */ /* v8 ignore next */
          let r = window.location.href; /* v8 ignore next */ /* v8 ignore next */
          (r.indexOf('#') != -1 && (r = r.substring(0, r.indexOf('#'))), /* v8 ignore next */ /* v8 ignore next */
            n.href.substring(0, r.length) == r && setTimeout(() => this.setActive(!1), 250)); /* v8 ignore next */ /* v8 ignore next */
        } /* v8 ignore next */ /* v8 ignore next */
      } /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  }; /* v8 ignore next */ /* v8 ignore next */
  var te; /* v8 ignore next */ /* v8 ignore next */
  try { /* v8 ignore next */ /* v8 ignore next */
    te = localStorage; /* v8 ignore next */ /* v8 ignore next */
  } catch { /* v8 ignore next */ /* v8 ignore next */
    te = { /* v8 ignore next */ /* v8 ignore next */
      getItem() { /* v8 ignore next */ /* v8 ignore next */
        return null; /* v8 ignore next */ /* v8 ignore next */
      }, /* v8 ignore next */ /* v8 ignore next */
      setItem() {} /* v8 ignore next */ /* v8 ignore next */
    }; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  var Q = te; /* v8 ignore next */ /* v8 ignore next */
  var ye = document.head.appendChild(document.createElement('style')); /* v8 ignore next */ /* v8 ignore next */
  ye.dataset.for = 'filters'; /* v8 ignore next */ /* v8 ignore next */
  var Y = class extends I { /* v8 ignore next */ /* v8 ignore next */
    constructor(e) { /* v8 ignore next */ /* v8 ignore next */
      (super(e), /* v8 ignore next */ /* v8 ignore next */
        (this.key = `filter-${this.el.name}`), /* v8 ignore next */ /* v8 ignore next */
        (this.value = this.el.checked), /* v8 ignore next */ /* v8 ignore next */
        this.el.addEventListener('change', () => { /* v8 ignore next */ /* v8 ignore next */
          this.setLocalStorage(this.el.checked); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        this.setLocalStorage(this.fromLocalStorage()), /* v8 ignore next */ /* v8 ignore next */
        (ye.innerHTML += `html:not(.${this.key}) .tsd-is-${this.el.name} { display: none; } /* v8 ignore next */ /* v8 ignore next */
`), /* v8 ignore next */ /* v8 ignore next */
        this.app.updateIndexVisibility()); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    fromLocalStorage() { /* v8 ignore next */ /* v8 ignore next */
      let e = Q.getItem(this.key); /* v8 ignore next */ /* v8 ignore next */
      return e ? e === 'true' : this.el.checked; /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    setLocalStorage(e) { /* v8 ignore next */ /* v8 ignore next */
      (Q.setItem(this.key, e.toString()), (this.value = e), this.handleValueChange()); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    handleValueChange() { /* v8 ignore next */ /* v8 ignore next */
      ((this.el.checked = this.value), /* v8 ignore next */ /* v8 ignore next */
        document.documentElement.classList.toggle(this.key, this.value), /* v8 ignore next */ /* v8 ignore next */
        this.app.filterChanged(), /* v8 ignore next */ /* v8 ignore next */
        this.app.updateIndexVisibility()); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  }; /* v8 ignore next */ /* v8 ignore next */
  var Z = class extends I { /* v8 ignore next */ /* v8 ignore next */
    constructor(e) { /* v8 ignore next */ /* v8 ignore next */
      (super(e), /* v8 ignore next */ /* v8 ignore next */
        (this.summary = this.el.querySelector('.tsd-accordion-summary')), /* v8 ignore next */ /* v8 ignore next */
        (this.icon = this.summary.querySelector('svg')), /* v8 ignore next */ /* v8 ignore next */
        (this.key = `tsd-accordion-${this.summary.dataset.key ?? this.summary.textContent.trim().replace(/\s+/g, '-').toLowerCase()}`)); /* v8 ignore next */ /* v8 ignore next */
      let n = Q.getItem(this.key); /* v8 ignore next */ /* v8 ignore next */
      ((this.el.open = n ? n === 'true' : this.el.open), /* v8 ignore next */ /* v8 ignore next */
        this.el.addEventListener('toggle', () => this.update())); /* v8 ignore next */ /* v8 ignore next */
      let r = this.summary.querySelector('a'); /* v8 ignore next */ /* v8 ignore next */
      (r && /* v8 ignore next */ /* v8 ignore next */
        r.addEventListener('click', () => { /* v8 ignore next */ /* v8 ignore next */
          location.assign(r.href); /* v8 ignore next */ /* v8 ignore next */
        }), /* v8 ignore next */ /* v8 ignore next */
        this.update()); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
    update() { /* v8 ignore next */ /* v8 ignore next */
      ((this.icon.style.transform = `rotate(${this.el.open ? 0 : -90}deg)`), /* v8 ignore next */ /* v8 ignore next */
        Q.setItem(this.key, this.el.open.toString())); /* v8 ignore next */ /* v8 ignore next */
    } /* v8 ignore next */ /* v8 ignore next */
  }; /* v8 ignore next */ /* v8 ignore next */
  function ge(t) { /* v8 ignore next */ /* v8 ignore next */
    let e = Q.getItem('tsd-theme') || 'os'; /* v8 ignore next */ /* v8 ignore next */
    ((t.value = e), /* v8 ignore next */ /* v8 ignore next */
      ve(e), /* v8 ignore next */ /* v8 ignore next */
      t.addEventListener('change', () => { /* v8 ignore next */ /* v8 ignore next */
        (Q.setItem('tsd-theme', t.value), ve(t.value)); /* v8 ignore next */ /* v8 ignore next */
      })); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  function ve(t) { /* v8 ignore next */ /* v8 ignore next */
    document.documentElement.dataset.theme = t; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  var Le; /* v8 ignore next */ /* v8 ignore next */
  function be() { /* v8 ignore next */ /* v8 ignore next */
    let t = document.getElementById('tsd-nav-script'); /* v8 ignore next */ /* v8 ignore next */
    t && (t.addEventListener('load', xe), xe()); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  async function xe() { /* v8 ignore next */ /* v8 ignore next */
    let t = document.getElementById('tsd-nav-container'); /* v8 ignore next */ /* v8 ignore next */
    if (!t || !window.navigationData) return; /* v8 ignore next */ /* v8 ignore next */
    let n = await (await fetch(window.navigationData)).arrayBuffer(), /* v8 ignore next */ /* v8 ignore next */
      r = new Blob([n]).stream().pipeThrough(new DecompressionStream('gzip')), /* v8 ignore next */ /* v8 ignore next */
      i = await new Response(r).json(); /* v8 ignore next */ /* v8 ignore next */
    ((Le = t.dataset.base + '/'), (t.innerHTML = '')); /* v8 ignore next */ /* v8 ignore next */
    for (let s of i) we(s, t, []); /* v8 ignore next */ /* v8 ignore next */
    (window.app.createComponents(t), window.app.showPage(), window.app.ensureActivePageVisible()); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  function we(t, e, n) { /* v8 ignore next */ /* v8 ignore next */
    let r = e.appendChild(document.createElement('li')); /* v8 ignore next */ /* v8 ignore next */
    if (t.children) { /* v8 ignore next */ /* v8 ignore next */
      let i = [...n, t.text], /* v8 ignore next */ /* v8 ignore next */
        s = r.appendChild(document.createElement('details')); /* v8 ignore next */ /* v8 ignore next */
      ((s.className = t.class ? `${t.class} tsd-index-accordion` : 'tsd-index-accordion'), /* v8 ignore next */ /* v8 ignore next */
        (s.dataset.key = i.join('$'))); /* v8 ignore next */ /* v8 ignore next */
      let o = s.appendChild(document.createElement('summary')); /* v8 ignore next */ /* v8 ignore next */
      ((o.className = 'tsd-accordion-summary'), /* v8 ignore next */ /* v8 ignore next */
        (o.innerHTML = /* v8 ignore next */ /* v8 ignore next */
          '<svg width="20" height="20" viewBox="0 0 24 24" fill="none"><use href="#icon-chevronDown"></use></svg>'), /* v8 ignore next */ /* v8 ignore next */
        Ee(t, o)); /* v8 ignore next */ /* v8 ignore next */
      let a = s.appendChild(document.createElement('div')); /* v8 ignore next */ /* v8 ignore next */
      a.className = 'tsd-accordion-details'; /* v8 ignore next */ /* v8 ignore next */
      let l = a.appendChild(document.createElement('ul')); /* v8 ignore next */ /* v8 ignore next */
      l.className = 'tsd-nested-navigation'; /* v8 ignore next */ /* v8 ignore next */
      for (let u of t.children) we(u, l, i); /* v8 ignore next */ /* v8 ignore next */
    } else Ee(t, r, t.class); /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  function Ee(t, e, n) { /* v8 ignore next */ /* v8 ignore next */
    if (t.path) { /* v8 ignore next */ /* v8 ignore next */
      let r = e.appendChild(document.createElement('a')); /* v8 ignore next */ /* v8 ignore next */
      ((r.href = Le + t.path), /* v8 ignore next */ /* v8 ignore next */
        n && (r.className = n), /* v8 ignore next */ /* v8 ignore next */
        location.pathname === r.pathname && r.classList.add('current'), /* v8 ignore next */ /* v8 ignore next */
        t.kind && /* v8 ignore next */ /* v8 ignore next */
          (r.innerHTML = `<svg width="20" height="20" viewBox="0 0 24 24" fill="none" class="tsd-kind-icon"><use href="#icon-${t.kind}"></use></svg>`), /* v8 ignore next */ /* v8 ignore next */
        (r.appendChild(document.createElement('span')).textContent = t.text)); /* v8 ignore next */ /* v8 ignore next */
    } else e.appendChild(document.createElement('span')).textContent = t.text; /* v8 ignore next */ /* v8 ignore next */
  } /* v8 ignore next */ /* v8 ignore next */
  G(X, 'a[data-toggle]'); /* v8 ignore next */ /* v8 ignore next */
  G(Z, '.tsd-index-accordion'); /* v8 ignore next */ /* v8 ignore next */
  G(Y, '.tsd-filter-item input[type=checkbox]'); /* v8 ignore next */ /* v8 ignore next */
  var Se = document.getElementById('tsd-theme'); /* v8 ignore next */ /* v8 ignore next */
  Se && ge(Se); /* v8 ignore next */ /* v8 ignore next */
  var je = new U(); /* v8 ignore next */ /* v8 ignore next */
  Object.defineProperty(window, 'app', { value: je }); /* v8 ignore next */ /* v8 ignore next */
  he(); /* v8 ignore next */ /* v8 ignore next */
  be(); /* v8 ignore next */ /* v8 ignore next */
})(); /* v8 ignore next */ /* v8 ignore next */
/*! Bundled license information: /* v8 ignore next */ /* v8 ignore next */
 /* v8 ignore next */ /* v8 ignore next */
lunr/lunr.js: /* v8 ignore next */ /* v8 ignore next */
  (** /* v8 ignore next */ /* v8 ignore next */
   * lunr - http://lunrjs.com - A bit like Solr, but much smaller and not as bright - 2.3.9 /* v8 ignore next */ /* v8 ignore next */
   * Copyright (C) 2020 Oliver Nightingale /* v8 ignore next */ /* v8 ignore next */
   * @license MIT /* v8 ignore next */ /* v8 ignore next */
   *) /* v8 ignore next */ /* v8 ignore next */
  (*! /* v8 ignore next */ /* v8 ignore next */
   * lunr.utils /* v8 ignore next */ /* v8 ignore next */
   * Copyright (C) 2020 Oliver Nightingale /* v8 ignore next */ /* v8 ignore next */
   *) /* v8 ignore next */ /* v8 ignore next */
  (*! /* v8 ignore next */ /* v8 ignore next */
   * lunr.Set /* v8 ignore next */ /* v8 ignore next */
   * Copyright (C) 2020 Oliver Nightingale /* v8 ignore next */ /* v8 ignore next */
   *) /* v8 ignore next */ /* v8 ignore next */
  (*! /* v8 ignore next */ /* v8 ignore next */
   * lunr.tokenizer /* v8 ignore next */ /* v8 ignore next */
   * Copyright (C) 2020 Oliver Nightingale /* v8 ignore next */ /* v8 ignore next */
   *) /* v8 ignore next */ /* v8 ignore next */
  (*! /* v8 ignore next */ /* v8 ignore next */
   * lunr.Pipeline /* v8 ignore next */ /* v8 ignore next */
   * Copyright (C) 2020 Oliver Nightingale /* v8 ignore next */ /* v8 ignore next */
   *) /* v8 ignore next */ /* v8 ignore next */
  (*! /* v8 ignore next */ /* v8 ignore next */
   * lunr.Vector /* v8 ignore next */ /* v8 ignore next */
   * Copyright (C) 2020 Oliver Nightingale /* v8 ignore next */ /* v8 ignore next */
   *) /* v8 ignore next */ /* v8 ignore next */
  (*! /* v8 ignore next */ /* v8 ignore next */
   * lunr.stemmer /* v8 ignore next */ /* v8 ignore next */
   * Copyright (C) 2020 Oliver Nightingale /* v8 ignore next */ /* v8 ignore next */
   * Includes code from - http://tartarus.org/~martin/PorterStemmer/js.txt /* v8 ignore next */ /* v8 ignore next */
   *) /* v8 ignore next */ /* v8 ignore next */
  (*! /* v8 ignore next */ /* v8 ignore next */
   * lunr.stopWordFilter /* v8 ignore next */ /* v8 ignore next */
   * Copyright (C) 2020 Oliver Nightingale /* v8 ignore next */ /* v8 ignore next */
   *) /* v8 ignore next */ /* v8 ignore next */
  (*! /* v8 ignore next */ /* v8 ignore next */
   * lunr.trimmer /* v8 ignore next */ /* v8 ignore next */
   * Copyright (C) 2020 Oliver Nightingale /* v8 ignore next */ /* v8 ignore next */
   *) /* v8 ignore next */ /* v8 ignore next */
  (*! /* v8 ignore next */ /* v8 ignore next */
   * lunr.TokenSet /* v8 ignore next */ /* v8 ignore next */
   * Copyright (C) 2020 Oliver Nightingale /* v8 ignore next */ /* v8 ignore next */
   *) /* v8 ignore next */ /* v8 ignore next */
  (*! /* v8 ignore next */ /* v8 ignore next */
   * lunr.Index /* v8 ignore next */ /* v8 ignore next */
   * Copyright (C) 2020 Oliver Nightingale /* v8 ignore next */ /* v8 ignore next */
   *) /* v8 ignore next */ /* v8 ignore next */
  (*! /* v8 ignore next */ /* v8 ignore next */
   * lunr.Builder /* v8 ignore next */ /* v8 ignore next */
   * Copyright (C) 2020 Oliver Nightingale /* v8 ignore next */ /* v8 ignore next */
   *) /* v8 ignore next */ /* v8 ignore next */
*/
