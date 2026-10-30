/**
 * @fileoverview jaxpr_parser module for converters.
 */
/**
 * JAX jaxpr JSON parser.
 */

export interface JaxprEqn {
  /** Property `primitive`. */
  primitive: string;
  /** Property `invars`. */
  invars: string[];
  /** Property `outvars`. */
  outvars: string[];
  /** Property `params`. */
  params: Record<string, object>;
}

/** Interface `Jaxpr` definition. */
export interface Jaxpr {
  /** Property `invars`. */
  invars: string[];
  /** Property `outvars`. */
  outvars: string[];
  /** Property `constvars`. */
  constvars: string[];
  /** Property `eqns`. */
  eqns: JaxprEqn[];
}

/**
 * Executes `parseJaxpr`.
 * @param content Parameter `content`.
 * @returns Result of `parseJaxpr`.
 */
export function parseJaxpr(content: string): Jaxpr {
  const data = JSON.parse(content) as Jaxpr;
  // Assume a very simplified JSON structure for jaxpr
  return data;
}
