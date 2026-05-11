/**
 * JAX jaxpr JSON parser.
 */

export interface JaxprEqn {
  primitive: string;
  invars: string[];
  outvars: string[];
  params: Record<string, object>;
}

export interface Jaxpr {
  invars: string[];
  outvars: string[];
  constvars: string[];
  eqns: JaxprEqn[];
}

export function parseJaxpr(content: string): Jaxpr {
  const data = JSON.parse(content) as Jaxpr;
  // Assume a very simplified JSON structure for jaxpr
  return data;
}
